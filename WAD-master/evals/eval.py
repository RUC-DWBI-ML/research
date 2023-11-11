import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import contrast.models.transform_layers as TL
from contrast.utils.utils import set_random_seed, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def knowledge_generation(P,model,unlabeled_loader,labeled_loader,group_loader,simclr_aug=None):

    kwargs = {
        'simclr_aug': simclr_aug,
        'layers': ['simclr']
    }
    label_u, feats_u, index_u = get_features(P, P.dataset, model, unlabeled_loader, **kwargs)
    label_l, feats_l, index_l = get_features(P, f'{P.dataset}_train', model, labeled_loader,**kwargs) 

    feats_group = []
    for i in range(len(group_loader)):
        label, feats, index = get_features(P, f'{P.dataset}_train', model, group_loader[i], **kwargs)  
        feats_group.append(feats)
    unlabeled_group_score = []
    for i in range(len(feats_group)):
        P.axis = []
        for f in feats_group[i]['simclr'].chunk(P.K_shift, dim=1):
            axis = f.mean(dim=1)
            P.axis.append(normalize(axis, dim=1).to(device))
        max_sim_group = get_scores(P, feats_u, label_l)
        unlabeled_group_score.append(max_sim_group)

    labels_logits = []
    weights = []
    for j in range(len(unlabeled_group_score[0])): # len(unlabeled_group_score[0]) indicates the number of unlabeled instances
        similarity_logits = []
        for i in range(len(unlabeled_group_score)): # len(unlabeled_group_score) indicates the categories
            similarity_logits.append(unlabeled_group_score[i][j])
        labels_logits.append(similarity_logits)
        sort_logit = sorted(similarity_logits)
        weights.append(sort_logit[-1] * (1 - sort_logit[-2] / sort_logit[-1]))

    return index_u, labels_logits, weights


def get_scores(P, feats_dict,labels):
    
    feats_sim = feats_dict['simclr'].to(device) # convert to gpu tensor
    N = feats_sim.size(0)
    max_sim = []
    for f_sim in feats_sim:
        f_sim = [normalize(f.mean(dim=0, keepdim=True), dim=1) for f in f_sim.chunk(P.K_shift)] 
        simi_score = 0
        for shi in range(P.K_shift):
            value_sim, indices_sim = ((f_sim[shi] * P.axis[shi]).sum(dim=1)).sort(descending=True)
            simi_score += value_sim.max().item()
        max_sim.append(simi_score)
    max_sim = torch.tensor(max_sim)

    assert max_sim.dim() == 1 and max_sim.size(0) == N  # (N)
    return max_sim.cpu()


def get_features(P, data_name, model,loader, prefix='',simclr_aug=None, layers=('simclr')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    feats_dict = dict()
    left = [layer for layer in layers if layer not in feats_dict.keys()]

    if len(left) > 0:
        labels,_feats_dict,index = _get_features(P, model, loader,simclr_aug, layers=left)
        for layer, feats in _feats_dict.items():
            path = prefix + '2' + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return labels,feats_dict,index


def _get_features(P, model, loader, simclr_aug=None,layers=('simclr')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    assert simclr_aug is not None

    labels = []
    index = []
    model.eval()
    feats_dict = {layer: [] for layer in layers}
    for i, (x, label, indices) in enumerate(loader):
        labels.extend(label)
        index.extend(indices)
        x = x.type(torch.FloatTensor)
        x = x.to(device)  # gpu tensor
        feats_batch = {layer: [] for layer in layers} # compute features in one batch
        if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
        else:
            x_t = x 
        x_t = simclr_aug(x_t)

        with torch.no_grad():
            kwargs = {layer: True for layer in layers}  # only forward selected layers
            _, output_aux = model(x_t, **kwargs)

        for layer in layers:
            feats = output_aux[layer].cpu()
            feats_batch[layer] += feats.chunk(P.K_shift)

        for key, val in feats_batch.items():
            feats_batch[key] = torch.stack(val, dim=1) 

        for layer in layers:
            feats_dict[layer] += [feats_batch[layer]]

    for key, val in feats_dict.items():
        feats_dict[key] = torch.cat(val, dim=0) 


    for key, val in feats_dict.items():
        N, T, d = val.size()  
        val = val.view(N, -1, P.K_shift, d) 
        val = val.transpose(2, 1) 
        val = val.reshape(N, T, d)  
        feats_dict[key] = val

    return labels,feats_dict,index




def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)



def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))


def test_classifier(P, model, loader, steps, marginal=False, logger=None):
    error_top1 = AverageMeter()
    error_calibration = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels,index) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(device), labels.to(device)

        if marginal:
            outputs = 0
            for i in range(4):
                rot_images = torch.rot90(images, i, (2, 3))
                _, outputs_aux = model(rot_images, joint=True)
                outputs += outputs_aux['joint'][:, P.n_classes * i: P.n_classes * (i + 1)] / 4.
        else:
            outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

        ece = ece_criterion(outputs, labels) * 100
        error_calibration.update(ece.item(), batch_size)

        if n % 100 == 0:
            log_('[Test %3d] [Test@1 %.3f] [ECE %.3f]' %
                 (n, error_top1.value, error_calibration.value))

    log_(' * [Error@1 %.3f] [ECE %.3f]' %
         (error_top1.average, error_calibration.average))

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error_top1.average, steps)
        logger.scalar_summary('eval/ece', error_calibration.average, steps)

    model.train(mode)

    return error_top1.average









