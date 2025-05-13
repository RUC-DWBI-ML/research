import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import math
import random
import logging
import numpy as np
import torchvision
import torch.nn as nn
from tqdm import tqdm
from dataset import *
from evaluate import *
from labeling import *
from loss import *
from argparse import ArgumentParser
from typing import Optional, Tuple
from synthetic_data_pipeline.generator import *
from synthetic_data_pipeline.dataset import *
import torch.nn.functional as F
from wideresnet import *
from torch.optim import Adam, SGD
from typing import List, Union
from torch.utils.data import Dataset, DataLoader,random_split,Subset
import torchvision.transforms as transforms


def save_model(model, optimizer, args):
    """
    Save the trained model and optimizer state.
    """
    save_path = f"./save_model/{args.data_name}_model_mis_{args.mismatch*10}_seed_{args.seed}.pth"
    save_dir = os.path.dirname(save_path) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
    logging.info(f"Model saved to {save_path}")



def train(args, unlabeled_data, unknown_data, known_data, indices_set, test_dataloader, closed_test_dataloader, pair_dataset, model, train_device):
    optimizer = Adam(model.parameters(), lr=args.lr)
    all_indices, unlabeled_indices, pseudo_known_indice, unknown_indices, dataset = indices_set
    pair_dataset.set_total_indices(unlabeled_indices)

    test_dataloader_known, test_dataloader_unknown, test_dataloader_new = test_dataloader

    # Initialize data loaders
    unlabeled_dataloader, known_dataloader, unknown_dataloader, unlabeled_iter, known_iter, unknown_iter = initialize_dataloaders(args, unlabeled_data, known_data, unknown_data, unlabeled_indices, pseudo_known_indice, unknown_indices, train_device)

    for i in range(args.epochs // args.select_epoch):
        batch_steps = max(len(d) // (args.batch_size) + 1 for d in (unlabeled_indices, pseudo_known_indice, unknown_indices) if d)
        iteration = batch_steps * args.select_epoch
        logging.info(f"unlabeled_indices:{len(unlabeled_indices)}, pseudo_known_indice:{len(pseudo_known_indice)}, unknown_indices:{len(unknown_indices)}, iteration:{iteration}")

        for iterstep in tqdm(range(iteration)):
            model.train()
            if unlabeled_dataloader is None:
                u_loss = 0
            else:
                u_data = next(unlabeled_iter)
                u_loss = compute_u_loss(u_data, pair_dataset, model, train_device, args)

            if known_dataloader is None:
                known_loss = 0
            else:
                k_data = next(known_iter)
                known_loss = compute_known_loss(k_data, pair_dataset, model, train_device, args)

            if unknown_dataloader is None:
                unknown_loss = 0
            else:
                uk_data = next(unknown_iter)
                unknown_loss = compute_unknown_loss(uk_data, pair_dataset, model, train_device, args)
            

            # Total loss
            loss = u_loss + known_loss + unknown_loss
            logging.info(f"u_loss:{u_loss}, known_loss:{known_loss}, unknown_loss:{unknown_loss}")

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging & evaluation
            if iterstep % 100 == 0:
                evaluate_model(test_dataloader_known, test_dataloader_unknown, test_dataloader_new, closed_test_dataloader, model, train_device, args)

        # Confidence-based labeling
        if unlabeled_dataloader is not None:
            model.eval()
            known_data, unlabeled_data, unknown_data, indices_set = confidence_based_labeling(args, model, known_data, unknown_data, unlabeled_data, indices_set, train_device, i, args.threshold)
            all_indices, unlabeled_indices, pseudo_known_indice, unknown_indices, dataset = indices_set
            unlabeled_dataloader, known_dataloader, unknown_dataloader, unlabeled_iter, known_iter, unknown_iter = initialize_dataloaders(args, unlabeled_data, known_data, unknown_data, unlabeled_indices, pseudo_known_indice, unknown_indices, train_device)


    # Final evaluation
    evaluate_model(test_dataloader_known, test_dataloader_unknown, test_dataloader_new, closed_test_dataloader, model, train_device, args)

    # Save model
    save_model(model, optimizer, args)

    print("Training over.")



def main():
    """
    Main function to parse arguments, load data, set up the model, and start training.
    """

    # Argument parsing
    parser = ArgumentParser()

    parser.add_argument("--train_device", default="cuda:0", type=str, help="Device for training.")
    parser.add_argument("--data_dir", default='/data', type=str, help="Data directory.")
    parser.add_argument("--load_path", default='/data', type=str, help="Path to load positive and negative instances.")
    parser.add_argument("--data_name", default='cifar10', type=str, help="Dataset name.")
    parser.add_argument("--epochs", default=400, type=int, help="Number of training epochs.")
    parser.add_argument("--threshold", default=0.98, type=float, help="Threshold for confidence labeling.")
    parser.add_argument("--num_classes", default=2, type=int, help="Number of target classes.")
    parser.add_argument("--known_class", default=list(range(0, 2)), type=list, help="Known classes.")
    parser.add_argument("--unknown_class", default=list(range(2, 8)), type=list, help="Unknown categories in unlabeled data.")
    parser.add_argument("--new_class", default=list(range(8, 10)), type=list, help="New categories in test data.")
    parser.add_argument("--test_size", default=(2000, 2000, 2000), type=tuple, help="Test set size (target, unknown, unknown).")
    parser.add_argument("--mismatch", default=0.6, type=float, help="Mismatch ratio.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--select_epoch", default=40, type=int, help="Epoch interval for data selection.")
    parser.add_argument("--lr", default=5e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--lambda1", default=1, type=float, help="Coefficient for the loss component addressing positive-negative instance discrimination.")
    parser.add_argument("--lambda2", default=2, type=float, help="Coefficient for the loss component addressing multi-class category classification.")
    parser.add_argument("--config", default="config.json", type=str, help="Config file.")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    args.known_class = config.get('known_class', [0, 1])
    args.unknown_class = config.get('unknown_class', [2, 3, 4, 5, 6, 7])
    args.new_class = config.get('new_class', [8, 9])
    args.test_size = config.get('test_size', (2000,2000,2000))


    # Log the arguments
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Arguments provided:")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")

    # Device configuration
    train_device = torch.device(args.train_device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {train_device}")

    # Dataset initialization
    train_set = CustomImageDataset(known_class=args.known_class, data_path=args.data_dir, dataset_name=args.data_name, download=False, is_train=True)
    test_set = CustomImageDataset(known_class=args.known_class, data_path=args.data_dir, dataset_name=args.data_name, download=False, is_train=False)
    logging.info("Dataset loaded.")

    # Process training and test data
    unlabeled_data, unknown_data, known_data, indices_set = mismatch_data(
        train_set, args.known_class, args.unknown_class, args.new_class,
        data_name=args.data_name, mismatch=args.mismatch, test_size=args.test_size, train=True,batch_size=args.batch_size
    )
    test_dataloader_known, test_dataloader_unknown, test_dataloader_new, closed_test_dataloader = mismatch_data(
        test_set, args.known_class, args.unknown_class, args.new_class,
        data_name=args.data_name, test_size=args.test_size, mismatch=args.mismatch, train=False,batch_size=args.batch_size
    )
    logging.info("Data loaders created.")

    # Create pair dataset
    pair_dataset = PositiveNegativePairDataset(generator=None, known_class=args.known_class, save_path=args.load_path)
    pair_dataset.load_generated_data()
    logging.info("Pair dataset created.")

    # Model initialization
    model = WideResNet_(depth=28, num_classes=args.num_classes, widen_factor=2, dropRate=0.0)
    model = model.to(train_device)
    logging.info("Model loaded.")

    # Training function
    train(args, unlabeled_data, unknown_data, known_data, indices_set, (test_dataloader_known, test_dataloader_unknown, test_dataloader_new), closed_test_dataloader, pair_dataset, model, train_device)

if __name__ == "__main__":
    main()
