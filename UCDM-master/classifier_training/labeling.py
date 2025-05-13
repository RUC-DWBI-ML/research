import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from typing import Tuple, List


def confidence_based_labeling(
    args,
    model,
    known_data,
    unknown_data,
    unlabeled_data,
    indices_set: Tuple[List[int], List[int], List[int], List[int], 'CustomImageDataset'],
    train_device,
    num: int,
    threshold: float = 0.98
):  
    """
    Select reliable unknown data based on model predictions and pseudo-labels.
    
    Args:
        args: Configuration arguments (batch size, etc.).
        model: The trained model used to generate predictions.
        known_data: The data labeled as known classes.
        unknown_data: The data labeled as unknown classes.
        unlabeled_data: Data that has not been labeled yet.
        indices_set: Indices set containing all indices, unlabeled indices, pseudo-label indices, unknown indices, and the dataset.
        train_device: Device to run the model on.
        num: The current iteration number.
        threshold: Threshold probability for selecting reliable samples.
    
    Returns:
        Updated known_data, unlabeled_data, and unknown_data.
    """
    # Unpacking indices set
    all_indices, unlabeled_indices, pseudo_known_indices, unknown_indices, dataset = indices_set
    logging.info(f"Iteration {num}: Selecting reliable unknown data.")

    # Initialize collections
    index_collect = []
    pseudo_label_collect = []
    known_index = []
    unknown_index = []
    
    # Prepare DataLoader
    unlabeled_dataloader = DataLoader(unlabeled_data, batch_size=args.batch_size, drop_last=False)
    model.eval()

    # Iterate over the unlabeled data
    for batch_idx, (inputs, _, indices) in enumerate(tqdm(unlabeled_dataloader, desc="Processing Unlabeled Data")):
        inputs, indices = inputs.to(train_device), indices.to(train_device)
        logits, detect, _ = model(inputs)

        closed_pred = F.softmax(logits, dim=1)
        detect = detect.view(detect.shape[0], 2, -1)
        open_pred = F.softmax(detect, dim=1)

        # Combining the predictions for closed and open sets
        o_other_pred = torch.prod(open_pred[:, 1, :], dim=-1, keepdim=True)
        o_known_pred = (1 - o_other_pred) * closed_pred
        other_driven_pred = torch.cat([o_known_pred, o_other_pred], dim=-1)
        
        k_known_pred = open_pred[:, 0, :] * closed_pred
        k_other_pred = 1 - torch.sum(k_known_pred, dim=1, keepdim=True)
        known_driven_pred = torch.cat([k_known_pred, k_other_pred], dim=-1)

        # Determine the predicted labels
        other_driven_confidence, other_driven_label = torch.max(other_driven_pred, dim=1)
        known_driven_confidence, known_driven_label = torch.max(known_driven_pred, dim=1)

        # select reliable samples
        mask = (other_driven_label == known_driven_label) & (other_driven_confidence > threshold) & (known_driven_confidence > threshold)
        confidence_pseudo_label = other_driven_label[mask]
        confidence_indices = indices[mask].cpu()

        # Collect the results
        pseudo_label_collect.extend(confidence_pseudo_label.tolist())
        index_collect.extend(confidence_indices.tolist())
       

        # Separate indices for known and unknown categories
        for pred, idx in zip(confidence_pseudo_label.tolist(), confidence_indices.tolist()):
            if pred < closed_pred.shape[1]:
                known_index.append(idx)
            else:
                unknown_index.append(idx)

    # Convert collected indices and labels to tensors
    index_collect_tensor = torch.tensor(index_collect).to(train_device)
    pseudo_label_collect_tensor = torch.tensor(pseudo_label_collect).to(train_device)
    known_index_ = torch.tensor(known_index).to(train_device)
    unknown_index_ = torch.tensor(unknown_index).to(train_device)

    classes = np.arange(logits.shape[1]+1)

    index_collect_, pseudo_label_collect_,known_index_,unknown_index_  = balanced_sample_pseudo_labels(classes, pseudo_label_collect_tensor,index_collect_tensor,known_index_,unknown_index_)

    # Set pseudo labels in dataset
    if len(index_collect_) != 0:
        dataset.set_pseudo_labels(torch.tensor(index_collect_, dtype=torch.long), torch.tensor(pseudo_label_collect_, dtype=torch.long))

        pseudo_known_indices.extend(known_index_)
        unknown_indices.extend(unknown_index_)

        unlabeled_indices = np.setdiff1d(unlabeled_indices, index_collect_, assume_unique=False).tolist()

        # Update DataLoader
        random.shuffle(pseudo_known_indices)
        random.shuffle(unknown_indices)

        if len(unlabeled_indices) != 0:
            unlabeled_data = Subset(dataset, unlabeled_indices)
        else:
            unlabeled_data = None

        if len(pseudo_known_indices) != 0:
            known_data = Subset(dataset, pseudo_known_indices)
        else:
            known_data = None

        if len(unknown_indices) != 0:
            unknown_data = Subset(dataset, unknown_indices)
        else:
            unknown_data = None

        logging.info(f"Updated DataLoader sizes -> Known: {len(pseudo_known_indices)}, Unlabeled: {len(unlabeled_indices)}, Unknown: {len(unknown_indices)}")

    return known_data, unlabeled_data, unknown_data, (
        all_indices,
        unlabeled_indices,
        pseudo_known_indices,
        unknown_indices,
        dataset
    )





def balanced_sample_pseudo_labels(
    classes,
    pseudo_label_collect_tensor,
    index_collect_tensor,
    known_index_,unknown_index_
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Randomly sample balanced pseudo-labels per class. Ensures that the number of samples per class is balanced
    according to the maximum class size.

    Args:
        pseudo_label_collect_tensor (torch.Tensor): Tensor containing the pseudo labels for all samples.
        index_collect_tensor (torch.Tensor): Tensor containing the indices of the samples.

    Returns:
        Tuple[List[int], List[int]]: Lists of sampled indices and their corresponding pseudo labels.
    """
    # Get the unique classes
    
    index_select = []
    pseudo_select = []
    
    # Count the number of samples per class
    each_class_num = []
    for cls in classes:
        cls_mask = (pseudo_label_collect_tensor == cls)
        if cls != len(classes)-1:
            each_class_num.append(torch.sum(cls_mask).item())


    # Find the maximum class count
    max_class_num = max(each_class_num)
    max_class_index = each_class_num.index(max_class_num)

    for cls in classes:
        # Mask for the current class
        cls_mask = (pseudo_label_collect_tensor == cls)
        cls_indices = index_collect_tensor[cls_mask]
        cls_pseudo_labels = pseudo_label_collect_tensor[cls_mask]
        num_selected = cls_indices.size(0)

        # Randomly sample or select all based on class size
        if num_selected > max_class_num:
            sampled_indices = random.sample(range(num_selected), max_class_num)
            sampled_cls_indices = cls_indices[sampled_indices]
            sampled_cls_labels = cls_pseudo_labels[sampled_indices]
        else:
            sampled_cls_indices = cls_indices
            sampled_cls_labels = cls_pseudo_labels

        index_select.extend(sampled_cls_indices.tolist())
        pseudo_select.extend(sampled_cls_labels.tolist())
        logging.info(f"Class {cls}: Pseudo Selected {num_selected} samples, Selected: {len(sampled_cls_indices)}")
    known_index_ = [x for x in known_index_ if x in index_select]
    unknown_index_ = [x for x in unknown_index_ if x in index_select]
    

    return index_select, pseudo_select,known_index_,unknown_index_

