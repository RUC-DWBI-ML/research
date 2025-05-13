from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging


def accuracy_with_count(y_true, y_pred):
    """
    Calculate the accuracy and the count of correct predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple: A tuple containing:
            - count (list): A list with two elements: the count of correct predictions
              and the total number of samples.
            - accuracy (float): The accuracy of the predictions as a fraction.
    """
    # Calculate the number of correct predictions
    correct_count = accuracy_score(y_true, y_pred, normalize=False)
    
    # Calculate accuracy
    accuracy = correct_count / len(y_true)
    
    # Return both count and accuracy
    count = [correct_count, len(y_true)]
    
    return count, accuracy



def closed_set_test(test_loader, model, train_device):
    """
    Evaluate the model on a closed-set classification task.

    Args:
        test_loader (DataLoader): The DataLoader for the test dataset.
        model (nn.Module): The trained model to evaluate.
        train_device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the accuracy and correct count.
            - "closed_accuracy": The accuracy on the closed-set test.
            - "correct_count": A list with two values: the correct count and total samples.
    """
    # Initialize lists to store predictions and true labels for closed set classification
    closed_pred = []
    closed_label = []

    # Set the model to evaluation mode and disable gradients
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            # Unpack batch data
            inputs, targets, _ = batch_data

            # Move data to the specified device
            inputs = inputs.to(train_device)
            targets = targets.to(train_device)

            # Get the model's predictions
            close_logits, logits_open, _ = model(inputs)

            # Compute softmax probabilities for the closed-set predictions
            score = F.softmax(close_logits, dim=-1)

            # Get the predicted class for each sample
            pred_close = score.argmax(dim=1)

            # Append predictions and true labels for later accuracy calculation
            closed_pred.extend(pred_close.cpu().numpy())
            closed_label.extend(targets.cpu().numpy())

        # Compute the accuracy for the closed-set classification
        correct_count, closed_accuracy = accuracy_with_count(closed_label, closed_pred)

        # Log the results
        logging.info(f"Closed-set Accuracy: {closed_accuracy:.4f}, Correct/Total: {correct_count[0]}/{correct_count[1]}")

    # Return the closed accuracy and correct count
    return closed_accuracy






def open_set_test(test_loader, model, train_device):
    """
    Evaluate the model on an open-set classification task.

    Args:
        test_loader (DataLoader): The DataLoader for the test dataset.
        model (nn.Module): The trained model to evaluate.
        train_device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: A tuple containing the following:
            - correct_count (list): A list with two values: the correct count and total samples.
            - open_accuracy (float): The accuracy on the open-set test.
    """
    # Initialize lists to store predictions and true labels for open-set classification
    all_pred = []
    all_label = []
    pred_prob = []

    # Set the model to evaluation mode and disable gradients
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            # Unpack batch data
            inputs, targets, _ = batch_data
            # Move data to the specified device
            inputs = inputs.to(train_device)
            targets = targets.to(train_device)

            # Get the model's predictions for closed and open-set logits
            close_logits, logits_open, _ = model(inputs)

            # Apply softmax to closed-set logits
            close_logits = F.softmax(close_logits, dim=-1)

            # Reshape and apply softmax to open-set logits
            logits_open = logits_open.view(logits_open.shape[0], 2, -1)
            logits_open_ = F.softmax(logits_open, dim=1)

            # Compute the other and known class scores
            other_score = torch.prod(logits_open_[:, 1, :], dim=-1, keepdim=True)
            known_score = (1 - other_score) * close_logits

            # Concatenate the known class score and open class score
            score = torch.cat([known_score, other_score], dim=-1)

            # Store prediction probabilities and open-set predictions
            pred_prob.append(score)
            pred_open = score.argmax(dim=1)  # Predicted class is the one with the highest score

            # Collect the predicted and true labels for accuracy calculation
            all_pred.extend(pred_open.cpu().numpy())
            all_label.extend(targets.cpu().numpy())

        # Calculate the accuracy and correct count for open-set classification
        correct_count, open_accuracy = accuracy_with_count(all_label, all_pred)
        
        # Log the results
        logging.info(f"Open-set Accuracy: {open_accuracy:.4f}, Correct/Total: {correct_count[0]}/{correct_count[1]}")

    # Return the correct count and open-set accuracy
    return correct_count, open_accuracy


def evaluate_model(test_dataloader_known, test_dataloader_unknown, test_dataloader_new, closed_test_dataloader, model, train_device, args):
    """
    Perform evaluation on the model at regular intervals.
    """
    model.eval()
    _, test_known_score = open_set_test(test_dataloader_known, model,  train_device)
    _, test_unknown_score = open_set_test(test_dataloader_unknown, model, train_device)
    _, test_new_score = open_set_test(test_dataloader_new, model,  train_device)
    closed_score = closed_set_test(closed_test_dataloader, model, train_device)

    print(f"test_known_score: {test_known_score}, test_unknown_score: {test_unknown_score}, test_new_score: {test_new_score}, closed_score: {closed_score}")


