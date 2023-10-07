
import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, num_epochs, train_dl, valid_dl, optimizer, loss_fn):
    """
    Trains a PyTorch model for a specified number of epochs using the given data loaders, optimizer, and loss function.

    Args:
        model (nn.Module): The PyTorch model to train.
        num_epochs (int): The number of epochs to train the model for.
        train_dl (DataLoader): The data loader for the training set.
        valid_dl (DataLoader): The data loader for the validation set.
        optimizer (Optimizer): The optimizer to use for training the model.
        loss_fn (Loss): The loss function to use for training the model.

    Returns:
        Tuple of four lists: loss_hist_train, accuracy_hist_train, loss_hist_valid, accuracy_hist_valid.
        loss_hist_train (list): The training loss for each epoch.
        accuracy_hist_train (list): The training accuracy for each epoch.
        loss_hist_valid (list): The validation loss for each epoch.
        accuracy_hist_valid (list): The validation accuracy for each epoch.
    """
    # Initialize lists to store loss and accuracy for each epoch
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0]*num_epochs
    accuracy_hist_valid = [0]*num_epochs

    # Loop through each epoch
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        # Loop through each batch in the training data loader
        for x_batch, y_batch in train_dl:
            # Move data to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Forward pass
            pred = model(x_batch)
            # Calculate loss
            loss = loss_fn(pred, y_batch)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()
            # Update loss and accuracy
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct =(torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        # Calculate average loss and accuracy for the epoch
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        # Set model to evaluation mode
        model.eval()
        # Disable gradient calculation
        with torch.no_grad():
            # Loop through each batch in the validation data loader
            for x_batch, y_batch in valid_dl:
                # Move data to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # Forward pass
                pred = model(x_batch)
                # Calculate loss
                loss = loss_fn(pred,y_batch)
                # Update loss and accuracy
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
            # Calculate average loss and accuracy for the epoch
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        # Print accuracy for the epoch
        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} (train) {accuracy_hist_valid[epoch]:.4f} (valid)')
    
    # Return loss and accuracy
    return loss_hist_train, accuracy_hist_train, loss_hist_valid, accuracy_hist_valid




def plot_graphs(hist):
    """
    Plots the loss and accuracy graphs for the training and validation sets.

    Args:
    hist (tuple): A tuple containing the loss and accuracy history for the training and validation sets.

    Returns:
    None
    """
    loss_hist_train, accuracy_hist_train, loss_hist_valid, accuracy_hist_valid = hist
    accuracy_hist_train = [x.item() for x in accuracy_hist_train]
    accuracy_hist_valid = [x.item() for x in accuracy_hist_valid]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(loss_hist_train, label='train', marker='o')
    plt.plot(loss_hist_valid, label='valid', marker='o')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(accuracy_hist_train, label='train', marker='o')
    plt.plot(accuracy_hist_valid, label='valid', marker='o')
    plt.legend()
    plt.show()



def visualize_predictions(model, data_loader, class_names, mean, std, num_samples=16):
    """
    Visualizes predictions made by a model on a test or validation set.

    Args:
    model (torch.nn.Module): The trained model.
    data_loader (torch.utils.data.DataLoader): The data loader for the test or validation set.
    class_names (list): A list of class names.
    mean (tuple): A tuple of mean values for each channel of the input images.
    std (tuple): A tuple of standard deviation values for each channel of the input images.
    num_samples (int): The number of samples to visualize. Default is 16.

    Returns:
    None
    """
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            if idx * images.shape[0] >= num_samples:
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for j in range(images.shape[0]):
                if (idx * images.shape[0] + j) >= num_samples:
                    break
                plt.subplot(4, 4, idx * images.shape[0] + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow((images[j].cpu().numpy().transpose(1, 2, 0) * std + mean).clip(0, 1))
                plt.xlabel(f'{class_names[labels[j]]} | {class_names[preds[j]]}')
    plt.show()
