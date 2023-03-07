# Importing libraries

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torchinfo import summary
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time

#print(dir(models))


# Setting device

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Initialize model with the best available weights
weights = models.VGG16_Weights.DEFAULT
model = models.vgg16(weights = weights).to(device)
print(model)

# Initialize the transforms
preprocess = weights.transforms()
preprocess

# Creating a class to create custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Importing training and test images and labels with preprocessing

train_img_loc = '/jet/home/gkant/train/train/'
train_labels = '/jet/home/gkant/train_labels.csv'
train_ds = CustomDataset(train_labels, train_img_loc, transform=preprocess)
train_loader = DataLoader(train_ds, batch_size=200, shuffle=True)
print(len(train_loader.dataset))

test_img_loc = '/jet/home/gkant/test/test/'
test_labels = '/jet/home/gkant/test_labels.csv'
test_ds = CustomDataset(test_labels, test_img_loc, transform=preprocess)
test_loader = DataLoader(test_ds, batch_size=200, shuffle=True)
print(len(test_loader.dataset))

hidden_loc = '/jet/home/gkant/hidden_test/hidden_test/'
hidden_labels = '/jet/home/gkant/NN_HW2/sample_submission.csv'
hidden_ds = CustomDataset(hidden_labels, hidden_loc, transform=preprocess)
hidden_loader = DataLoader(hidden_ds)
print(len(hidden_loader.dataset))


'''
# Print a summary using torchinfo. This is to view input and outputs of each layer and modify as required according to dataset
summary(model=model, 
        input_size=(200, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 
'''
# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False
    
# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# We have 2 classes. VGG16 is trained on ImageNet dataset which has 1000 classes
output_shape = 2

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=25088, 
                    out_features=output_shape,
                    bias=True)).to(device)

'''
# Print summary after freezing the features and changing the output classifier layer
summary(model, 
        input_size=(200, 3, 224, 224),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
'''

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create training loop
def train_loop(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

#Create test loop
def test_loop(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Combine training and test steps in a single loop

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"epoch":[],
               "train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Model on target device
    model.to(device)

    # Loop through training and testing steps for epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_loop(model=model,
                                          dataloader=train_loader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_loop(model=model,
                                        dataloader=test_loader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["epoch"].append(epoch)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

# Set the random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
start_time = time.time()

# Setup training and save the results
results = train(model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=20,
                device=device)

end_time = time.time()
print(f"Total training time: {(end_time-start_time)/60:.3f} minutes")

# Save dictionary- results as csv

results_df = pd.DataFrame(results)
results_df.to_csv('/jet/home/gkant/NN_HW2/CNN_v2_results.csv', index=False)

# Prediction for hidden test images

def hidden_test_prediction(model, dataloader, device):

    model.eval()
    pred_labels = []
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through hidden_test dataloader
        for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # Make predictions
                ht_pred_logits = model(X)
                ht_pred_labels = ht_pred_logits.argmax(dim=1)
                
                for i in ht_pred_labels:
                    pred_labels.append(i.item())
                    
    return pred_labels
    
ht_pred = hidden_test_prediction(model, hidden_loader, device)

# Importing input file for submission
hidden_test_df = pd.read_csv('/jet/home/gkant/NN_HW2/sample_submission.csv')

# Obtaining final file for submission
ht_pred_sub = pd.DataFrame({'img_name':hidden_test_df['img_name'],
                        'label': ht_pred})
ht_pred_sub.to_csv('/jet/home/gkant/NN_HW2/CNN_v2_ht_pred_sub.csv', index=False)
