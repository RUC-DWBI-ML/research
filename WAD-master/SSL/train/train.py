import random
import torch
import os, sys
import matplotlib
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from datasets import rectify_labels
matplotlib.use('Agg')
sys.path.append(os.getcwd())


class Solver:
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.mse_loss_false_reduce = nn.MSELoss(reduce=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_false_reduce = nn.CrossEntropyLoss(reduce=False)

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, index in dataloader:
                    yield img, label, index
        else:
            while True:
                for img, _, index in dataloader:
                    yield img, index

    def read_data_supervised(self, dataloader, ssl_labeled_indices, ssl_pseudo_label,labels=True):
        if labels:
            while True:
                for img, label, index in dataloader:
                    pseudo_label, _ = self.alignment(index,ssl_labeled_indices, ssl_pseudo_label)
                    yield img, torch.tensor(pseudo_label), index
        else:
            while True:
                for img, _, index in dataloader:
                    yield img, index

    def alignment(self, base_index, index, label, weights=[],weight=False):
        
        align_weight = []
        align_label = []
        if weight:
            for j in range(len(base_index)):
                i = index.index(base_index[j])
                align_label.append(torch.argmax(torch.tensor(label[i]).type(torch.FloatTensor),dim=0))
                align_weight.append(weights[i])
            align_weight = torch.tensor(align_weight)
            
        else:
            for j in range(len(base_index)):
                i = index.index(base_index[j])
                align_label.append(label[i])
        return align_label, align_weight
            

    
    def backbone_classifier(self, classifier, labeled_dataloader, test_dataloader,unlabeled_dataloader,label_logits,pseudo_index,weights,ssl_labeled_indice, ssl_current_label):

        train_iterations = self.args.epochs
        labeled_data = self.read_data_supervised(labeled_dataloader,ssl_labeled_indice, ssl_current_label)
        unlabeled_data = self.read_data(unlabeled_dataloader)
        optimizer = torch.optim.Adam(classifier.params(), lr=5e-4)

        if self.args.cuda:
            classifier = classifier.cuda()

        for idx, iter_count in enumerate(tqdm(range(train_iterations))):

            classifier.train()
            
            labeled_imgs, labels, index = next(labeled_data)
            labeled_imgs = labeled_imgs.type(torch.FloatTensor)
            correct_labels = rectify_labels(labels, self.args)
            unlabeled_imgs, unlabeled_labels, unlabeled_index = next(unlabeled_data)
            unlabeled_imgs = unlabeled_imgs.type(torch.FloatTensor)
            # Label alignment
            pseudo_label, weight_ = self.alignment(unlabeled_index, pseudo_index, label_logits, weights, weight=True)

            if labeled_imgs.size(0) < 2:
                labeled_imgs, labels, index = next(labeled_data)
                labeled_imgs = labeled_imgs.type(torch.FloatTensor)
                correct_labels = rectify_labels(labels, self.args)
            if unlabeled_imgs.size(0) < 2:
                unlabeled_imgs, unlabeled_labels, unlabeled_index = next(unlabeled_data)
                unlabeled_imgs = unlabeled_imgs.type(torch.FloatTensor)
                pseudo_label, weight_ = self.alignment(unlabeled_index, pseudo_index, label_logits, weights, weight=True)

            pseudo_label = torch.stack(pseudo_label)
            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                correct_labels = correct_labels.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                pseudo_label = pseudo_label.cuda()
                weight_ = weight_.cuda()

            labeled_preds_logits = classifier(labeled_imgs)
            labeled_loss = self.ce_loss(labeled_preds_logits, correct_labels)
            unlabeled_preds_logits = classifier(unlabeled_imgs)
            unlabeled_loss_weight = weight_ * self.ce_loss_false_reduce(unlabeled_preds_logits, pseudo_label)
            unlabeled_loss = torch.mean(unlabeled_loss_weight)
            task_loss = labeled_loss + unlabeled_loss

            optimizer.zero_grad()
            task_loss.backward()
            optimizer.step()

            if (iter_count + 1) % len(labeled_dataloader) == 0:
                accuracy_test = self.Test(classifier, test_dataloader, what='test')

            if iter_count % 100 == 0:
                print('Current training iteration: {}'.format(iter_count))
                print('Current task model loss: {:.4f}'.format(task_loss.item()))

            if (iter_count < 5) or (iter_count % 50 == 49):
                print("classifier loss is:",task_loss.item())

        accuracy_test = self.Test(classifier, test_dataloader, what='test')
        return classifier, accuracy_test



    def Test(self, classifier, test_dataloader, what='train'):

        classifier.eval()
        total, correct = 0, 0
        for id_l, (imgs, labels, indices) in enumerate(tqdm(test_dataloader)):
            imgs = imgs.type(torch.FloatTensor)
            if self.args.cuda:
                imgs = imgs.cuda()
            class_preds = classifier(imgs)
            class_preds_s = F.softmax(class_preds, dim=1)
            correct_label = rectify_labels(labels, self.args)
            preds = torch.argmax(class_preds_s, dim=1).cpu().numpy()
            correct += accuracy_score(correct_label, preds, normalize=False)
            total += imgs.size(0)
        accuracy = correct / total * 100
        print("-{}- correct is {}".format(what, correct))
        print("-{}- total is {}".format(what, total))
        print("-{}- accuracy is {}".format(what, accuracy))

        return accuracy


    def calculate_sample_predict(self, classifier_ssl, ssl_unlabeled_dataloader):
        classifier_ssl.eval()
        all_preds_logits = []
        all_indices = []
        all_labels = []
        for idx1, (imgs, labels, indices) in enumerate(tqdm(ssl_unlabeled_dataloader)):
            imgs = imgs.type(torch.FloatTensor)
            if self.args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                class_preds = classifier_ssl(imgs)
            all_indices.extend(indices)
            all_preds_logits.extend(class_preds)
            all_labels.extend(labels)

        return all_indices, all_labels, all_preds_logits


    def calculate_loss_logits_pseudo(self, label_logits, pseudo_index, predict_logit, predict_index, true_label,alpha,u_number):
     
            ground_ = torch.argmax(torch.tensor(label_logits).type(torch.FloatTensor), dim=1) #return the pseudo label that provided 
            predict_ = torch.stack([torch.tensor(item) for item in predict_logit]).cuda()
            arr_s = np.array(pseudo_index)
            arr_p = np.array(predict_index)
            sp_index = (arr_s == arr_p[:, None]).argmax(1)
            arr_ground = np.array(ground_)
            pseudo_label = list(arr_ground[sp_index])
            ground_pseudo = torch.tensor(arr_ground[sp_index]).cuda()

            loss = self.ce_loss_false_reduce(predict_,ground_pseudo)  # return the gap between predict logit and one hot label
            loss = loss * (-1)

            _, query_inside = torch.topk(loss, int(alpha * u_number))
            query_inside = query_inside.cpu().data

            reliable_label = np.asarray(pseudo_label)[query_inside]
            reliable_indices = np.asarray(predict_index)[query_inside]

            return reliable_indices, reliable_label
