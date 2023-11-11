import torch
import os
import random
import argument
import numpy as np
import contrast.models.classifier as C
import torch.utils.data as data
from evals.eval import knowledge_generation
from SSL.train.custom_datasets import *
from SSL.train.train import Solver
from SSL.wideresnet import WideResNet
from SSL.transform import transform
from datasets import polynomial_decay, Get_dataloader, shuffles, Get_group_index_train
from datasets import Get_group_index, get_contrastive_dataset,Get_initial_dataset, Get_mismatch_unlabeled_dataset

def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # args.ood_layer = args.ood_layer[0]  ### only use one ood_layer while training
    kwargs = {'pin_memory': False, 'num_workers': 4}

    if args.dataset == 'cifar10':

        train_set_contrastive, test_set_contrastive, image_size, n_classes = get_contrastive_dataset(args, dataset=args.dataset)
        train_dataset_ssl = Cifar10(args.data_path)
        test_dataset_ssl = Cifar10_test(args.data_path)
        args.image_size = image_size
        args.n_classes = n_classes
        args.num_images = 50000
        args.initial_budget = 800
        args.batch_size_classifier = 32
        args.target_list = [0, 1]
        args.untarget_list = [2, 3, 4, 5, 6, 7, 8, 9]
        args.target_number = 2

    elif args.dataset == 'cifar100':

        train_set_contrastive, test_set_contrastive, image_size, n_classes = get_contrastive_dataset(args, dataset=args.dataset)
        train_dataset_ssl = Cifar100(args.data_path)
        test_dataset_ssl = Cifar100_test(args.data_path)
        args.image_size = image_size
        args.n_classes = n_classes
        args.num_images = 50000
        args.initial_budget = 800
        args.batch_size_classifier = 32
        args.target_list = [3, 42, 43, 88, 97, 15, 19, 21, 32, 39, 35, 63, 64, 66, 75, 37, 50, 65, 74, 80]
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
        args.target_number = 20
        raise NotImplementedError

    """
    Processing for Contrastive Dataset
    """
    initial_con_dataset, initial_con_index = Get_initial_dataset(train_set_contrastive, args.target_list, args.initial_budget) # contrstive dataset for labeled data
    unlabeled_con_dataset, unlabeled_con_index = Get_mismatch_unlabeled_dataset(train_set_contrastive, initial_con_index, args.target_list,args.mismatch, args.num_images) # contrastive dataset for unlabeled data
    con_labeled_dataloader = Get_dataloader(train_set_contrastive, initial_con_index, args) # contrastive labeled dataloader
    con_unlabeled_dataloader = Get_dataloader(train_set_contrastive, unlabeled_con_index, args) # contrastive unlabeled dataloader
    group_index = Get_group_index(train_set_contrastive,initial_con_index, args)
    group_loader = Get_dataloader(train_set_contrastive, initial_con_index, args, group_index=group_index, group=True)

    """
    Processing for SSL dataset
    """
    initial_ssl_dataset, initial_ssl_indices = Get_initial_dataset(train_dataset_ssl, args.target_list, args.initial_budget) 
    ssl_current_label = [train_dataset_ssl[initial_ssl_indices[i]][1] for i in range(len(initial_ssl_indices))]
    initial_ssl_indices = shuffles(initial_ssl_indices)
    ssl_current_label = shuffles(ssl_current_label)
    ssl_labeled_dataloader = Get_dataloader(train_dataset_ssl, initial_ssl_indices, args, ssl=True)
    unlabeled_ssl_dataset, ssl_unlabeled_index = Get_mismatch_unlabeled_dataset(train_dataset_ssl,initial_ssl_indices, args.target_list,args.mismatch, args.num_images)
    ssl_test_indices = [test_dataset_ssl[i][2] for i in range(len(test_dataset_ssl)) if test_dataset_ssl[i][1] in args.target_list]
    ssl_test_dataloader = Get_dataloader(test_dataset_ssl, ssl_test_indices, args, ssl=True)
    """
    Some initial settings
    """
    train_indices = range(len(train_dataset_ssl))
    train_dataloader = Get_dataloader(train_dataset_ssl, train_indices, args, ssl=True)
    args.cuda = args.cuda and torch.cuda.is_available()  # use gpu
    solver = Solver(args, train_dataloader)
    unlabeled_ssl_dataset, ssl_unlabeled_index = Get_mismatch_unlabeled_dataset(train_dataset_ssl,initial_ssl_indices, args.target_list,args.mismatch, args.num_images)
    ssl_all_indices = ssl_unlabeled_index + list(initial_ssl_indices)
    ssl_labeled_indice = list(initial_ssl_indices)

    """
    Get Contrastive model
    """
    simclr_aug = C.get_simclr_augmentation(args, image_size=args.image_size).to(device)
    args.shift_trans, args.K_shift = C.get_shift_module(args, eval=True)
    args.shift_trans = args.shift_trans.to(device)
    if args.mode == 'eval':
        model_contrastive = C.get_classifier(args.model, n_classes=args.n_classes).to(device)
        model_contrastive = C.get_shift_classifer(model_contrastive, args.K_shift).to(device)
        assert args.load_teacher_path is not None
        checkpoint_senmatic = torch.load(args.load_teacher_path)
        args.no_strict = False
        model_contrastive.load_state_dict(checkpoint_senmatic, strict=not args.no_strict)
    simclr_aug = C.get_simclr_augmentation(args, image_size=args.image_size).to(device)

    """
    Get SSL model
    """
    classifiers_ssl = WideResNet(widen_factor=2, n_classes=len(args.target_list), transform_fn=transform(), seed=args.seed).to(device)


    """
    Distillation Process
    """
    for i in range(1,args.split+2):
        
        """
        Training the target model
        """
        ssl_unlabeled_indices = np.setdiff1d(ssl_all_indices,ssl_labeled_indice) 
        ssl_unlabeled_indices = shuffles(ssl_unlabeled_indices)
        ssl_unlabeled_dataloader = Get_dataloader(train_dataset_ssl, ssl_unlabeled_indices, args, ssl=True)

        # generate the pseudo labels and weights
        with torch.no_grad():
            index, pseudo_label_logits,weights = knowledge_generation(args,model_contrastive, con_unlabeled_dataloader, con_labeled_dataloader, group_loader, simclr_aug=simclr_aug)

        # train the target classifier
        classifiers_ssl, accuracy_ssl = solver.backbone_classifier(classifiers_ssl, ssl_labeled_dataloader, ssl_test_dataloader,ssl_unlabeled_dataloader, pseudo_label_logits, index, weights, ssl_labeled_indice, ssl_current_label)
        print("The accuracy in the {}-th iteration is {}".format(i, accuracy_ssl))


        # Obtain the predict logits of unlabeled instances
        predict_indices, true_label, predict_logit = solver.calculate_sample_predict(classifiers_ssl, ssl_unlabeled_dataloader)
        alpha = polynomial_decay(i, args.alpha, 0, args.split,2)
        print("The alpha in the {}-th iteration is {}".format(i+1, alpha))

        # Select some instances according to loss and then add them to the labeled data.
        ssl_query_indices, ssl_query_label_pseudo = solver.calculate_loss_logits_pseudo(pseudo_label_logits, index, predict_logit, predict_indices, true_label, alpha, len(ssl_unlabeled_indices))

        # transform the pseudo labels to one of target categories, for example, 0->25
        ssl_query_label = []
        for k in range(len(ssl_query_label_pseudo)):
            ssl_query_label.append(args.target_list[ssl_query_label_pseudo[k]])

        ssl_labeled_indice = ssl_labeled_indice + list(ssl_query_indices)   

        # generate the new labeled dataloader for ssl which contains ground truth labels and pseudo labels
        ssl_current_label = ssl_current_label + list(ssl_query_label)
        ssl_labeled_indice = shuffles(ssl_labeled_indice)
        ssl_current_label = shuffles(ssl_current_label)  
        ssl_labeled_dataloader = Get_dataloader(train_dataset_ssl, ssl_labeled_indice, args, ssl=True)      

        """
        Update the dataloader for contrastive learning
        """
        initial_con_index = initial_con_index + list(ssl_query_indices)
        split_group_index = Get_group_index_train(list(ssl_query_label), list(ssl_query_indices), args)
        for k in range(len(args.target_list)):
            group_index[k] += split_group_index[k]
        group_loader = Get_dataloader(train_set_contrastive, initial_con_index, args, group_index=group_index, group=True)
        unlabeled_con_index = list(np.setdiff1d(list(unlabeled_con_index), list(ssl_query_indices)))
        con_unlabeled_dataloader = Get_dataloader(train_set_contrastive, unlabeled_con_index, args)
        con_labeled_dataloader = Get_dataloader(train_set_contrastive, initial_con_index, args)



    print("The training is over.")


if __name__ == '__main__':
    args = argument.parse_args()
    main(args)