from synthetic_data_pipeline.dataset import *
import random
import math
import logging
from torch.utils.data import Subset, DataLoader

def create_dataloader(args, data) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.

    Parameters:
    - args (Namespace): The argument namespace containing the configuration (e.g., batch_size).
    - data (Dataset): The dataset to be loaded.
    Returns:
    - DataLoader: The DataLoader object configured with the specified arguments.
    """
    if not hasattr(args, 'batch_size'):
        raise ValueError("The 'args' object must have a 'batch_size' attribute.")
    
    # Create the DataLoader
    dataloader = DataLoader(
        data, 
        batch_size=args.batch_size, 
        drop_last=False, 
        shuffle=True
    )
    
    return dataloader

def read_data(dataloader):
    """
    A generator function that yields batches of images, labels, and indices from the DataLoader.

    This function is designed to continuously provide data from the DataLoader in an infinite loop.
    
    Parameters:
    - dataloader (DataLoader): The DataLoader object providing the data in batches.
    
    Yields:
    - tuple: A tuple of (img, label, index) for each batch of data from the DataLoader.
    """
    if dataloader is None:
        raise ValueError("The provided dataloader is None. Please provide a valid DataLoader.")
    
    while True:
        for img, label, index in dataloader:
            yield img, label, index


def initialize_dataloaders(args, unlabeled_data, known_data, unknown_data, unlabeled_indices, pseudo_known_indice, unknown_indices, train_device):
    """
    Initialize dataloaders for unlabeled, known, and unknown datasets.
    """
    unlabeled_dataloader, known_dataloader, unknown_dataloader = None, None, None
    unlabeled_iter,known_iter,unknown_iter = None, None, None

    if unlabeled_data is not None:
        unlabeled_dataloader = create_dataloader(args, unlabeled_data)
        unlabeled_iter = read_data(unlabeled_dataloader)

    if known_data is not None:
        known_dataloader = create_dataloader(args, known_data)
        known_iter = read_data(known_dataloader)

    if unknown_data is not None:
        unknown_dataloader = create_dataloader(args, unknown_data)
        unknown_iter = read_data(unknown_dataloader)

    return unlabeled_dataloader, known_dataloader, unknown_dataloader, unlabeled_iter,known_iter,unknown_iter


def mismatch_data(dataset, known_class, unknown_class, new_class, data_name="cifar10", mismatch=0.6, test_size=(0, 0, 0), train=True, batch_size=32):
    """
    Process dataset with a mismatch ratio for both training and testing phases.
    
    Args:
        dataset: The dataset to process.
        known_class: List of known class labels.
        unknown_class: List of unknown class labels.
        new_class: List of new class labels.
        data_name: Name of the dataset (default is "cifar10").
        mismatch: The mismatch ratio for unlabeled data (default is 0.6).
        test_size: Tuple of test set sizes for known, unknown, and new classes.
        train: Boolean flag to specify whether to process the training or test set.
        batch_size: Batch size for DataLoader.
    
    Returns:
        For training:
            - unlabeled_data: The subset containing labeled and unlabeled data.
            - unknown_data, target_data: None (can be populated if needed).
            - indices_set: A tuple containing all relevant indices.
        
        For testing:
            - test_known_dataloader, test_unknown_dataloader, test_new_dataloader, test_closed_dataloader: DataLoader objects for the test sets.
    """
    # Divide the dataset into indices based on the categories
    known_indices, unknown_indices, new_indices, all_indices = group_dataset_by_class(
        dataset, known_class, unknown_class, new_class
    )

    if train:
        logging.info("Processing train set...")
        
        # Calculate the number of unlabeled unknown samples
        num_unlabeled_unknown_samples = math.ceil(len(known_indices) / (1 - mismatch)) - len(known_indices)
        
        # Sample the required number of unlabeled unknown indices
        unlabeled_unknown_indices = split_indices(unknown_indices, train_size=num_unlabeled_unknown_samples)
        
        # Combine known and unlabeled unknown indices
        unlabeled_indices = list(known_indices) + list(unlabeled_unknown_indices)
        logging.info(f"Mismatch: {mismatch}, unlabeled known indices: {len(known_indices)}, "
                     f"unlabeled unknown indices: {len(unlabeled_unknown_indices)}, unlabeled data: {len(unlabeled_indices)}.")
        
        # Shuffle the indices
        random.shuffle(unlabeled_indices)
        
        # Create a Subset for unlabeled data
        unlabeled_data = Subset(dataset, unlabeled_indices)

        # Return the processed training data and indices
        unknown_data, target_data = None, None  # These can be populated if needed for future use
        unknown_indice, pseudo_target_indice = [], []  # Placeholder for indices, if needed
        return unlabeled_data, unknown_data, target_data, (all_indices, unlabeled_indices, pseudo_target_indice, unknown_indice, dataset)
    
    else:
        logging.info("Processing test set...")

        # Extract the required sample sizes for test subsets
        test_known, test_unknown, test_new = test_size
        
        # Sample indices for each test category
        random.seed(42)
        test_known_indices = random.sample(known_indices, test_known)
        random.seed(42)
        test_unknown_indices = random.sample(unknown_indices, test_unknown)
        random.seed(42)
        test_new_indices = random.sample(new_indices, test_new)

        # Create DataLoader for each test set
        test_known_data = Subset(dataset, test_known_indices)
        test_known_dataloader = DataLoader(test_known_data, batch_size=batch_size, shuffle=True)

        test_unknown_data = Subset(dataset, test_unknown_indices)
        test_unknown_dataloader = DataLoader(test_unknown_data, batch_size=batch_size, shuffle=True)

        test_new_data = Subset(dataset, test_new_indices)
        test_new_dataloader = DataLoader(test_new_data, batch_size=batch_size, shuffle=True)

        test_closed_data = Subset(dataset, known_indices)
        test_closed_dataloader = DataLoader(test_closed_data, batch_size=batch_size, shuffle=True)

        logging.info(f"Mismatch: {mismatch}, known: {len(test_known_indices)}, unknown: {len(test_unknown_indices)}, "
                     f"new: {len(test_new_indices)}, closed: {len(known_indices)}")
        
        # Return test set DataLoaders
        return test_known_dataloader, test_unknown_dataloader, test_new_dataloader, test_closed_dataloader



def split_indices(indices, train_size=None):
    """
    Shuffle the indices and return a subset of the specified size using random.sample.

    Args:
        indices: List of indices to split.
        train_size: Number of indices to select from the shuffled indices.
    
    Returns:
        train_indices: A list containing the randomly selected indices.
    """
    random.seed(42)  # Set the random seed for reproducibility
    random.shuffle(indices)
    # Use random.sample to select a subset of indices
    random.seed(42)
    train_indices = random.sample(list(indices), train_size)
    
    return train_indices
