import torch
import argument
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from contrast.training.scheduler import GradualWarmupScheduler
import contrast.models.classifier as C
import torch.optim.lr_scheduler as lr_scheduler
from contrast.utils.utils import load_checkpoint
from contrast.utils.utils import Logger
from contrast.utils.utils import save_checkpoint
from contrast.utils.utils import save_linear_checkpoint
from evals.eval import test_classifier
from contrast.utils.utils import set_random_seed
from contrast import setup
from SSL.train.custom_datasets import *
from datasets import *


def main(args):

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cifar10':

        train_set, test_set, image_size, n_classes = get_dataset(args, dataset=args.dataset)

        args.image_size = image_size
        args.n_classes = n_classes
        args.num_images = 50000
        args.initial_budget = 800
        args.num_classes = 10
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 32
        args.target_list = [0,1]
        args.untarget_list = [2,3,4,5,6,7,8,9]
        args.target_number = 2

    elif args.dataset == 'cifar100':

        train_set, test_set, image_size, n_classes = get_dataset(args, dataset=args.dataset)

        args.image_size = image_size
        args.n_classes = n_classes
        args.num_images = 50000
        args.initial_budget = 800
        args.num_classes = 20
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 32
        args.target_list = [3, 42, 43, 88, 97, 15, 19, 21, 32, 39, 35, 63, 64, 66, 75, 37, 50, 65, 74, 80]
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
        args.target_number = 20

    else:
        raise NotImplementedError



    """
    Build Contrastive Mismatch Dataset
    """
    initial_dataset, select_L_index = Get_initial_dataset(train_set, args.target_list,args.initial_budget)
    contrast_dataset, contrast_index = Get_mismatch_contrast_dataset(train_set, select_L_index, args.target_list,args.mismatch, args.num_images)
    contrastive_index = list(contrast_index)
    contrastive_sampler = data.sampler.SubsetRandomSampler(contrastive_index)  # make indices initial to the samples
    contrastive_loader = data.DataLoader(train_set, sampler=contrastive_sampler,batch_size=args.batch_size, drop_last = True)

    """
    Initialize the contrastive model
    """
    simclr_aug = C.get_simclr_augmentation(args, image_size=args.image_size).to(device)
    args.shift_trans, args.K_shift = C.get_shift_module(args, eval=True)
    args.shift_trans = args.shift_trans.to(device)
    model = C.get_classifier(args.model, n_classes=args.n_classes).to(device)
    model = C.get_shift_classifer(model, args.K_shift).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.weight_decay)
        lr_decay_gamma = 0.1
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, betas=(.9, .999), weight_decay=args.weight_decay)
        lr_decay_gamma = 0.3
    else:
        raise NotImplementedError()

    if args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * args.epochs), int(0.75 * args.epochs)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()


    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=args.warmup,after_scheduler=scheduler)

    if args.resume_path is not None:
        resume = True
        model_state, optim_state, config = load_checkpoint(args.resume_path, mode='last')
        model.load_state_dict(model_state, strict=not args.no_strict)
        optimizer.load_state_dict(optim_state)
        start_epoch = config['epoch']
        best = config['best']
    else:
        resume = False
        start_epoch = 1
        best = 100.0
    
    """
    Train the contrastive model
    """

    train, fname = setup(args.mode, args)

    logger = Logger(fname, ask=not resume, local_rank=args.local_rank)
    logger.log(args)
    logger.log(model)
    logger.logdir = args.logdir
    linear = model.linear
    linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + 1):
        logger.log_dirname(f"Epoch {epoch}")
        model.train()

        kwargs = {}
        kwargs['linear'] = linear
        kwargs['linear_optim'] = linear_optim
        kwargs['simclr_aug'] = simclr_aug

        train(args, epoch, model, criterion, optimizer, scheduler_warmup, contrastive_loader, logger=logger, **kwargs)

        model.eval()
        if epoch % args.save_step == 0 and args.local_rank == 0:
            save_states = model.state_dict()
            save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
            save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

        if epoch % args.error_step == 0 and ('sup' in args.mode):
            error = test_classifier(args, model, contrastive_loader, epoch, logger=logger)

            is_best = (best > error)
            if is_best:
                best = error

            logger.scalar_summary('eval/best_error', best, epoch)
            logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))

    print("-save model-")
    save_states = model.state_dict()
    save_checkpoint(800, save_states, optimizer.state_dict(), logger.logdir)
    save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)
    print("-Training is over-")



if __name__ == '__main__':
    args = argument.parse_args()
    set_random_seed(0)
    main(args)
