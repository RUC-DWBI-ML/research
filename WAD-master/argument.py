from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of WAD')
    parser.add_argument('--data_path', type=str, default='~/data', help='Path to where the data is')
    parser.add_argument('--dataset', help='Dataset',choices=['cifar10', 'cifar100'], type=str)
    parser.add_argument('--model', help='Model',choices=['resnet18', 'resnet18_imagenet'], type=str)
    parser.add_argument('--mode', help='Training mode',default='teacher', type=str)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',default=128, type=int)
    parser.add_argument('--shift_trans_type', help='shifting transformation type', default='none',choices=['rotation', 'cutperm', 'none'], type=str)
    parser.add_argument("--local_rank", type=int, default=0, help='Local rank for distributed learning')
    parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',default=0.08, type=float)
    parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',action='store_true')
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',default=None, type=str)    
    parser.add_argument('--load_path', help='Path to the loading checkpoint',default='./result_model', type=str)
    parser.add_argument('--load_teacher_path', help='Path to the loading checkpoint',default='./result_model/semantic', type=str)
    parser.add_argument('--logdir', help='Path to the loading checkpoint',default='./',type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',default=None, type=str)
    parser.add_argument('--error_step', help='Epoch steps to compute errors',default=5, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save models',default=10, type=int)
    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',default=1500, type=int)
    parser.add_argument('--mismatch', help='mismatch',default=0.8, type=float)
    parser.add_argument('--optimizer', help='Optimizer',choices=['sgd', 'adam', 'lars'],default='sgd', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',choices=['step_decay', 'cosine'],default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs', default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',default=128, type=int)
    parser.add_argument('--con_batch_size', help='Batch size for test loader',default=100, type=int)
    parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',default=1.0, type=float)
    parser.add_argument('--temperature', help='Temperature for similarity',default=0.5, type=float)
    parser.add_argument("--print_score", help='',action='store_true')
    parser.add_argument("--save_score", help='',action='store_true')
    parser.add_argument("--split", help='the times add the samples to labeled data',default=5, type=int)
    parser.add_argument('--alpha', help='top alpha instances as confident ones', default=0.1, type=float)
    parser.add_argument('--network', help='NetWork',choices=['resnet18', 'wideresnet_28_2'],type=str)
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log',help='Final performance of the models will be saved with this name')
    parser.add_argument('--seed', type=int, default=1,help='0 is initial, others is query times')


    if default:
        return parser.parse_args('')  
    else:
        return parser.parse_args()
