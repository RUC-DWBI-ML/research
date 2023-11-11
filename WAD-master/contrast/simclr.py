import time
import torch.optim
import contrast.models.transform_layers as TL
from contrast.training.contrastive_loss import get_similarity_matrix, NT_xent
from contrast.utils.utils import AverageMeter, normalize
from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['sim'] = AverageMeter()

    check = time.time()
    for n, (images, labels,index) in enumerate(loader):
        model.train()
        count = n 
        data_time.update(time.time() - check)
        check = time.time()
        images = images.type(torch.FloatTensor)
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images_pair = hflip(images.repeat(2, 1, 1, 1))  # 2B with hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
            images_pair = torch.cat([images1, images2], dim=0)  # 2B

        labels = labels.to(device)
        labels = rectify_labels(labels, P)

        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda

        loss = loss_sim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']
        batch_time.update(time.time() - check)
        losses['sim'].update(loss_sim.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['sim'].value))

        check = time.time()
    log_('[DONE] [Time %.3f] [Data %.3f] [LossSim %f]' %
         (batch_time.average, data_time.average, losses['sim'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
