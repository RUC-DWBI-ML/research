import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split,Subset
import random
import math
import torchvision
from torch import nn
from tqdm import tqdm
import time
import torch.nn.functional as F

from tqdm import tqdm
import time
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
try:
    from .generator import *
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from generator import *

import gzip
from torch.utils.data import DataLoader, DistributedSampler
from multiprocessing import Lock


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S' 
)
logger = logging.getLogger(__name__) 


def load_transform(image_size=(32, 32)):
    """
    Creates image transformations for both training and testing datasets.

    Parameters:
    - image_size (tuple): The target size of the images for resizing. Default is (32, 32).

    Returns:
    - tuple: A tuple containing the training and testing transformations.
    """
    # Training transformations: Resize, Random Horizontal Flip, ToTensor, Normalize
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Testing transformations: Resize, ToTensor, Normalize
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, test_transform



def load_dataset(data_path, dataset_name, download=False):
    """
    This function loads the specified dataset for training and testing.

    Parameters:
    - data_path: The path to the data directory.
    - dataset_name: The name of the dataset to load ('cifar10' or 'cifar100').
    - download: Whether to download the dataset if not already present (default: False).

    Returns:
    - train_set: The training dataset.
    - test_set: The test dataset.
    - image_size: The shape of the images in the dataset.
    - n_classes: The number of classes in the dataset.
    """
    
    # Define default image size
    image_size = (32, 32)
    
    # Get transformation functions
    train_transform, test_transform = load_transform(image_size=image_size)

    # Load CIFAR-10 dataset
    if dataset_name == 'cifar10':
        n_classes = 10
        train_set = datasets.CIFAR10(data_path, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(data_path, train=False, download=download, transform=test_transform)

    # Load CIFAR-100 dataset
    elif dataset_name == 'cifar100':
        n_classes = 100
        train_set = datasets.CIFAR100(data_path, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(data_path, train=False, download=download, transform=test_transform)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    return train_set, test_set, image_size, n_classes




class PositiveNegativePairDataset(Dataset):
    """
    A dataset for managing positive-negative pairs for training.
    
    Attributes:
    - shared_dict (dict): A dictionary shared among processes for storing positive-negative pairs.
    - shared_list (list): A list of created indices shared across processes.
    - positive_negative_pairs (dict): Dictionary of positive-negative pairs.
    - created_indices (list): List of indices that have been processed.
    - save_path (str): Path to save/load the dataset.
    - all_data_created (bool): Flag to indicate if all data has been generated.
    - total_indices (list): List of all indices to generate.
    - local_pairs (dict): Local store for newly created pairs.
    - local_indices (list): Local store for newly created indices.
    - generator (object): Generator responsible for creating positive-negative pairs.
    - known_class (list): List of target categories for generating pairs.
    """
    
    def __init__(self, generator, known_class, total_indices=None, 
                 save_path="/data/positive_negative_pairs.pt.gz", 
                 shared_dict=None, shared_list=None, shared_value=None):
        """
        Initialize the dataset with necessary attributes.

        Parameters:
        - generator: The generator object to generate positive-negative pairs.
        - known_class: List of target categories for pair generation.
        - total_indices: The total indices of the dataset.
        - save_path: The file path to save the dataset.
        - shared_dict: Shared dictionary to store generated pairs.
        - shared_list: Shared list to store indices of generated data.
        - shared_value: Shared value for flagging data generation completion.
        """
        
        self.shared_dict = shared_dict if shared_dict is not None else {}
        self.shared_list = shared_list if shared_list is not None else []
        self.positive_negative_pairs = shared_dict if shared_dict is not None else {}
        self.created_indices = shared_list if shared_list is not None else []
        self.save_path = save_path
        self.all_data_created = shared_value.value if shared_value is not None else False
        self.total_indices = total_indices
        self.local_pairs = {}
        self.local_indices = []
        self.generator = generator
        self.known_class = known_class
        
        if self.total_indices is not None:
            logging.info(f"Ready to generate {len(self.total_indices)} samples.")
    
    def set_total_indices(self, total_indices):
        """
        Set the total indices for the dataset.

        Parameters:
        - total_indices: The total indices of the dataset.
        """
        self.total_indices = total_indices
        logging.info(f"Ready to generate {len(self.total_indices)} samples.")
        logging.info(f"Initialized dataset: Created indices: {len(self.created_indices)}, Data generated: {self.check_all_data_created()}.")

    def check_all_data_created(self):
        """
        Check if all data has been generated.

        Returns:
        - bool: True if all data has been created, False otherwise.
        """
        all_created = len(self.created_indices) == len(self.total_indices)
        logging.info(f"All data created? {all_created} [Created/Total]: {len(self.created_indices)}/{len(self.total_indices)}")
        self.all_data_created = all_created
        return all_created

    def save_generated_data(self):
        """
        Save the generated positive-negative pairs to a file.
        """
        with gzip.open(self.save_path, 'wb') as f:
            torch.save(dict(self.positive_negative_pairs), f)
        logging.info("Dataset has been saved!")

    def load_generated_data(self):
        """
        Load previously saved positive-negative pairs from a file.
        """
        if os.path.exists(self.save_path):
            with gzip.open(self.save_path, 'rb') as f:
                self.shared_dict = torch.load(f)
                self.shared_list = list(self.shared_dict.keys())
                self.positive_negative_pairs = self.shared_dict
                self.created_indices = self.shared_list
                logging.info(f"Loaded dataset from {self.save_path}: Created indices: {len(self.created_indices)}.")
        else:
            logging.info("No saved data found. Generating new data...")
            self.positive_negative_pairs = self.shared_dict
            self.created_indices = self.shared_list

    def update_positive_negative_pairs(self):
        """
        Update the dataset with new positive-negative pairs and indices.
        """
        self.created_indices.extend(self.local_indices)
        self.positive_negative_pairs.update(self.local_pairs)
        self.local_pairs = {}
        self.local_indices = []
        logging.info(f"Updated local pairs: {len(self.local_pairs)}, Total pairs: {len(self.positive_negative_pairs)}")
        self.check_all_data_created()

    def local_store_pairs(self, index_list, positives, negatives, labels):
        """
        Store generated positive-negative pairs locally before updating the main dataset.

        Parameters:
        - index_list: List of indices for which pairs are generated.
        - positives: The generated positive samples.
        - negatives: The generated negative samples.
        - labels: The corresponding labels for the pairs.
        """
        self.local_indices.extend(index_list)
        positives_ = list(positives.chunk(len(index_list)))
        negatives_ = list(negatives.chunk(len(index_list)))
        labels_ = list(labels.chunk(len(index_list)))

        for i in range(len(index_list)):
            index = int(index_list[i])
            self.local_pairs[index] = {'positives': positives_[i], 'negatives': negatives_[i], 'labels': labels_[i]}
        logging.info(f"Local pairs stored: {len(self.local_pairs)}, Total pairs: {len(self.positive_negative_pairs)}")

    def generate_and_store_data(self, index_list, inputs, labels=None):
        """
        Generate positive-negative pairs for given indices and store them locally.

        Parameters:
        - index_list: The list of indices to generate pairs for.
        - inputs: The input data corresponding to the indices.
        - labels: The labels for the inputs (optional).
        """
        if not self.all_data_created:
            mask = torch.tensor([int(item) not in self.positive_negative_pairs.keys() for item in index_list], dtype=bool)
            mask_inputs = inputs[mask]
            mask_index = index_list[mask]

            if labels is not None:
                mask_labels = labels[mask]

            if mask_index.shape[0] != 0:
                if labels is not None:
                    uncon_inputs, con_inputs, g_labels = self.generator.generate(mask_inputs, label_map(self.known_class, mask_labels))
                else:
                    uncon_inputs, con_inputs, g_labels = self.generator.generate(mask_inputs)

                self.local_store_pairs(mask_index, con_inputs, uncon_inputs, g_labels)
                if len(self.local_pairs) > 40:
                    self.update_positive_negative_pairs()

    def __getitem__(self, index_list):
        """
        Retrieve positive and negative samples for the given indices.

        Parameters:
        - index_list: List of indices to fetch positive-negative pairs.

        Returns:
        - tuple: A tuple containing negatives, positives, and labels for the given indices.
        """
        positives = []
        negatives = []
        pn_labels = []

        for index_ in index_list:
            index = int(index_)
            if index not in self.positive_negative_pairs:
                logging.warning(f"Key {index} not found in positive-negative pairs.")

            positives.append(self.positive_negative_pairs[index]['positives'])
            negatives.append(self.positive_negative_pairs[index]['negatives'])
            pn_labels.append(self.positive_negative_pairs[index]['labels'])

        positives_ = torch.cat(positives, dim=0)
        negatives_ = torch.cat(negatives, dim=0)
        pn_labels_ = torch.cat(pn_labels, dim=0)

        return negatives_, positives_, pn_labels_







class CustomImageDataset(Dataset):
    def __init__(self, known_class=None, data_path: str = '/data', dataset_name: str = 'cifar10', download: bool = False, is_train: bool = True):
        """
        Custom Dataset to handle image data and associated pseudo-labels.

        Parameters:
        - data_path (str): Path to the dataset storage location.
        - dataset_name (str): The name of the dataset (e.g., 'cifar10').
        - download (bool): Whether to download the dataset if not present.
        - is_train (bool): Whether to load the training or testing split of the dataset.
        """
        self.is_train = is_train
        self.known_class = known_class
        train_set, test_set, image_size, n_classes = load_dataset(data_path, dataset_name, download)
        self.dataset = train_set if is_train else test_set
        self.labels = None  # Placeholder for pseudo-labels
        self.use_pseudo_label = False  # Flag to indicate if pseudo-labels are used

        logger.info(f"Initialized CustomImageDataset with {'train' if is_train else 'test'} split, size: {len(self.dataset)}")

    def __getitem__(self, index: int, only_label: bool = False):
        """
        Retrieve the data and label at the specified index.

        Parameters:
        - index (int): Index of the sample.
        - only_label (bool): If True, returns only the label.

        Returns:
        - Tuple: (image, label, index) if `only_label` is False, else (index, label).
        """
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, label = self.dataset[index]

        if only_label:
            return index, label

        # Use pseudo-labels if enabled
        if self.use_pseudo_label and self.labels is not None:
            label = self.labels[index].item()

        else:
            xlabel = label
            label = label_map(self.known_class, [label])[0]
        

        return data, label, torch.tensor(index, dtype=torch.long)


    def initialize_labels(self) -> None:
        """
        Initialize the labels by copying the true labels from the dataset.
        """
        logger.info("Initializing labels from the dataset.")
        self.labels = torch.tensor([label for _, label in self.dataset], dtype=torch.long)
        self.labels = torch.tensor(list(label_map(self.known_class, self.labels)))

    def set_pseudo_labels(self, indices: List[int], pseudo_labels: torch.Tensor) -> None:
        """
        Set pseudo-labels for specified indices.

        Parameters:
        - indices (List[int]): List of indices to update with pseudo-labels.
        - pseudo_labels (torch.Tensor): Tensor containing the pseudo-labels.
        """
        if self.labels is None:
            self.initialize_labels()

        if not isinstance(pseudo_labels, torch.Tensor):
            pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)

        if len(indices) != len(pseudo_labels):
            logger.error("Indices and pseudo_labels must have the same length.")
            raise ValueError("Indices and pseudo_labels must have the same length.")

        for idx, pseudo_label in zip(indices, pseudo_labels):
            if idx < 0 or idx >= len(self.labels):
                logger.warning(f"Index {idx} is out of bounds for labels of size {len(self.labels)}.")
                continue

            self.labels[idx] = pseudo_label

        self.use_pseudo_label = True
        logger.info("Pseudo labels are now enabled.")


    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.dataset)





def process_data_distributed(dataset, known_classes: List[int], unknown_classes: List[int], new_classes: List[int], batch_size: int=30,
                             world_size: int = None, rank: int = None) -> Tuple[DataLoader, List[int]]:
    """
    Process the dataset for distributed training. This function divides the dataset into known, unknown, and new classes, 
    and prepares the data for loading across multiple GPUs in a distributed setting.

    Parameters:
    - dataset: The dataset to process.
    - known_classes (List[int]): Known classes in training.
    - unknown_classes (List[int]): Unknown classes in training.
    - new_classes (List[int]): Classes unseen in training.
    - world_size (int): The number of GPUs or nodes in the distributed setup.
    - rank (int): The rank of the current process in the distributed setup.

    Returns:
    - DataLoader for distributed training.
    - List of all indices (known, unknown, new) used for sampling.
    """
    
    # Divide the dataset into known, unknown, and new classes
    known_indices, unknown_indices, new_indices, _ = group_dataset_by_class(dataset, known_classes, unknown_classes, new_classes)
    logging.info(f"Known classes: {len(known_indices)}, Close Unknown: {len(unknown_indices)}, Open Unknown: {len(new_indices)}")

    # Combine all indices (known, unknown, new classes)
    all_indices = list(known_indices) + list(unknown_indices) + list(new_indices)

    # If no distributed setup is provided, return indices directly
    if world_size is None and rank is None:
        return all_indices

    # Create a subset of the dataset based on the indices
    data_subset = Subset(dataset, all_indices)
    
    # Create DistributedSampler for splitting data across GPUs
    data_sampler = DistributedSampler(data_subset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    
    # Create DataLoader for distributed training
    dataloader = DataLoader(data_subset, batch_size=batch_size, sampler=data_sampler)
    
    logging.info(f"Total data size: {len(all_indices)}, Distributed sampler: num_replicas={world_size}, rank={rank}")

    return dataloader, all_indices



def group_dataset_by_class(dataset, known_class, unknown_class, new_class):
    """
    Divides the dataset into known, unknown, and new class indices.

    Parameters:
    - dataset: The dataset to divide.
    - known_class: List of indices for known classes.
    - unknown_class: List of indices for unknown classes.
    - new_class: List of indices for new classes.

    Returns:
    - Known class indices, unknown class indices, new class indices, all indices.
    """
    # Get all categories and indices from the dataset
    all_indices = np.array([i for i in range(len(dataset))])
    all_categories = np.array([dataset.__getitem__(i, only_label=True)[1] for i in range(len(dataset))])

    # Convert class lists to arrays for efficient comparison
    known_class_arr = np.array(known_class)
    unknown_class_arr = np.array(unknown_class)
    new_class_arr = np.array(new_class)

    # Get indices for each category
    known_indices = all_indices[np.isin(all_categories, known_class_arr)]
    unknown_indices = all_indices[np.isin(all_categories, unknown_class_arr)]
    new_indices = all_indices[np.isin(all_categories, new_class_arr)]

    return list(known_indices), list(unknown_indices), list(new_indices), list(all_indices)



def label_map(known_class, labels):
    """
    Maps each label in 'labels' to the corresponding index in 'known_class'.
    If the label is not found in 'known_class', it is mapped to the index
    of the "unknown" class, which is len(known_class).

    Parameters:
    - known_class: List of known class labels.
    - labels: Tensor or list of labels to be mapped.

    Returns:
    - A tensor of mapped labels.
    """
    # Create a mapping of labels to their index in known_class
    label_to_index = {label: idx for idx, label in enumerate(known_class)}

    # Map the labels
    mapped_labels = [label_to_index.get(label, len(known_class)) for label in labels]

    return torch.tensor(mapped_labels, dtype=torch.long)




