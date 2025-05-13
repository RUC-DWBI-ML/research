import torch
import numpy as np
import torch.nn.functional as F

loss_function = torch.nn.CrossEntropyLoss()

def detect_loss(
    positive_open_pred_: torch.Tensor,
    negative_open_pred_: torch.Tensor,
    closed_class_pred: torch.Tensor,
    labels: torch.Tensor,
    train_device: torch.device,
    lambda1: float = 1,
    lambda2: float = 2,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute the detection loss and class loss for an open-set classification problem.
    
    Parameters:
    - positive_open_pred_ (torch.Tensor): Predictions for positive open classes.
    - negative_open_pred_ (torch.Tensor): Predictions for negative open classes.
    - closed_class_pred (torch.Tensor): Predictions for closed classes (fully supervised).
    - labels (torch.Tensor): True labels.
    - train_device (torch.device): The device to perform computation on.
    - class_strength (float): The weight for the class loss term.
    - epsilon (float): A small constant to avoid log(0) errors.

    Returns:
    - torch.Tensor: The total loss combining detection and classification losses.
    """
    
    # Apply softmax to open-set predictions (reshape to 2xN format for binary detection)
    negative_open_pred = F.softmax(negative_open_pred_.view(negative_open_pred_.shape[0], 2, -1), dim=1)
    positive_open_pred = F.softmax(positive_open_pred_.view(positive_open_pred_.shape[0], 2, -1), dim=1)

    # Initialize detection label tensor with zeros
    detect_label = torch.zeros_like(negative_open_pred, device=train_device)

    # Set the detection labels for the current true class
    detect_label[torch.arange(detect_label.shape[0]), 0, labels] = 1
    detect_label[torch.arange(detect_label.shape[0]), 1, labels] = 1

    # Compute detection loss by calculating cross-entropy loss for both positive and negative open predictions
    positive_loss = torch.sum(-torch.log(positive_open_pred[:, 0, :] + epsilon) * detect_label[:, 0, :]) / (torch.sum(detect_label[:, 0, :]) + epsilon)
    negative_loss = torch.sum(-torch.log(negative_open_pred[:, 1, :] + epsilon) * detect_label[:, 1, :]) / (torch.sum(detect_label[:, 1, :]) + epsilon)

    detect_loss = positive_loss + negative_loss

    # Classification loss: cross-entropy for closed classes and positive open class predictions
    class_loss = loss_function(closed_class_pred, labels) + loss_function(positive_open_pred_.view(positive_open_pred_.shape[0], 2, -1)[:, 0, :], labels)

    # Total loss is the combination of detection loss and class loss
    total_loss = lambda1 * detect_loss + lambda2 * class_loss

    return total_loss


def compute_u_loss(unlabeled_data, pair_dataset, model, train_device, args):
    """
    Compute the loss for unlabeled data.
    """
    u_inputs, _, u_index = unlabeled_data
    negative_u_inputs, positive_u_inputs, u_labels = pair_dataset[u_index]
    negative_u_inputs, positive_u_inputs, u_labels = (
        negative_u_inputs.to(train_device),
        positive_u_inputs.to(train_device),
        u_labels.to(train_device),
    )
    
    u_loss = detect_loss(
        model(positive_u_inputs)[1], model(negative_u_inputs)[1], model(positive_u_inputs)[0], u_labels,
        train_device, lambda1=args.lambda1, lambda2=args.lambda2
    )
    return u_loss


def compute_known_loss(known_data, pair_dataset, model, train_device, args):
    """
    Compute the loss for labeled (known) data.
    """
    t_inputs, t_labels, t_index = known_data
    negative_t_inputs_, positive_t_inputs, _ = pair_dataset[t_index]
    
    negative_t_inputs = negative_t_inputs_.view(
        t_inputs.shape[0], len(args.known_class), *negative_t_inputs_.shape[1:]
    )[torch.arange(t_inputs.shape[0]), t_labels].to(train_device)

    t_inputs, t_labels, positive_t_inputs, negative_t_inputs = (
        t_inputs.to(train_device),
        t_labels.to(train_device),
        positive_t_inputs.to(train_device),
        negative_t_inputs.to(train_device)
    )

    t_loss = detect_loss(
        model(t_inputs)[1], model(negative_t_inputs)[1], model(t_inputs)[0], t_labels,
        train_device, lambda1=args.lambda1, lambda2=args.lambda2
    )
    return t_loss


def compute_unknown_loss(unknown_data, pair_dataset, model, train_device, args):
    """
    Compute the loss for unknown data.
    """
    unknown_inputs, _, unknown_index = unknown_data
    negative_unknown_inputs, positive_unknown_inputs_, unknown_labels_ = pair_dataset[unknown_index]
    unknown_labels = unknown_labels_.view(unknown_inputs.shape[0], len(args.known_class))

    random_indices = torch.randint(0, unknown_labels.shape[1], (unknown_labels.shape[0],))
    unk_label = unknown_labels[torch.arange(unknown_labels.shape[0]), random_indices]
    
    positive_unknown_inputs = positive_unknown_inputs_.view(
        unknown_inputs.shape[0], len(args.known_class), *positive_unknown_inputs_.shape[1:]
    )[torch.arange(unknown_inputs.shape[0]), unk_label].to(train_device)
    unk_label = unk_label.to(train_device)

    unknown_inputs, negative_unknown_inputs, positive_unknown_inputs, unknown_labels_ = (
        unknown_inputs.to(train_device),
        negative_unknown_inputs.to(train_device),
        positive_unknown_inputs.to(train_device),
        unknown_labels_.to(train_device),
    )

    unknown_loss = detect_loss(
        model(positive_unknown_inputs)[1], model(unknown_inputs)[1],
        model(positive_unknown_inputs)[0], unk_label, train_device, lambda1=args.lambda1, lambda2=args.lambda2
    )
    return unknown_loss
