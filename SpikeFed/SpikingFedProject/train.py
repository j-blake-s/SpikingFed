import torch
import os
from torch.utils.data import DataLoader
from utils.training import train, test
from model.snn import SNN
import numpy as np
from torch.utils.data import Dataset
import copy


# Args
from utils.argparser import get_args
args = get_args()


# Model
from model.snn import SNN
####################
#model = SNN().to(args.device)
model = SNN().to(args.device)
global_weights = model.state_dict()  # Store initial model weights
###################
print(f'Parameters: {model.params():,}')


# Data
from utils.dvs_data import DvsGesture
#######################################################################################
# Load dataset and create user partitions
dataset_train, dict_users_train = DvsGesture(os.path.join(args.data_path, "train.npz"), num_users=args.num_users)
dataset_test, dict_users_test = DvsGesture(os.path.join(args.data_path, "test.npz"), num_users=args.num_users)
test_dataset = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
#######################################################################################

class DatasetSplit(Dataset):
    """Class to split dataset for each federated learning client."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label

class LocalUpdate(object):
    def __init__(self, idx, model, dataset_train, dataset_test, idxs, idxs_test, args):
        self.idx = idx
        self.device = args.device
        self.model = model
        self.lr = args.lr
        self.local_epochs = args.local_epochs  # Number of local epochs
        self.loss_func = slayer.loss.SpikeRate(true_rate=0.5, false_rate=0.05, reduction='mean').to(self.device)

        # Create local data loaders for this client
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=args.batch_size, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=args.batch_size, shuffle=True)

    def train(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        epoch_loss = 0
        correct = 0
        total = 0

        for _ in range(self.local_epochs):  
            for images, labels in self.ldr_train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred = slayer.classifier.Rate.predict(outputs)
                # pred = outputs.argmax(dim=1, keepdim=True)  # Get predictions
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total += labels.size(0)

        avg_loss = epoch_loss / len(self.ldr_train)  
        accuracy = 100.0 * correct / total

        return self.model.state_dict(), avg_loss, accuracy  # Return weights, loss, accuracy


def FedAvg(w_locals):
    """Federated Averaging: Compute the mean of all client model updates."""
    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg
#############################################################################

# from utils.data import load_data
# dataset_train, dataset_test,_,_ = load_data(args.data_path, T=16)


"""data_loader = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True)"""


# Learning Tools
import lava.lib.dl.slayer as slayer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
error = slayer.loss.SpikeRate(true_rate=0.5, false_rate=0.05, reduction='mean').to(args.device)
classer = slayer.classifier.Rate.predict


# # Run
# torch.autograd.set_detect_anomaly(True)
"""for epoch in range(args.epochs):
  print(" "*50,end="\r")
  print(f'Epoch [{epoch+1}/{args.epochs}]')
  train_acc = train(model, data_loader, optimizer, error, classer, args)
  test_acc = test(model, data_loader_test, classer, args)
  print(f'\033[F\rEpoch [{epoch+1}/{args.epochs}] Training: {train_acc:.2%}\tValidation: {test_acc:.2%}              ')"""
####################################################################
# Store accuracy & loss for training & testing
train_loss_collect, test_loss_collect = [], []
train_acc_collect, test_acc_collect = [], []

for epoch in range(args.epochs):
    print(f"Epoch [{epoch+1}/{args.epochs}]")

    w_locals = []  # Store weights from local clients
    loss_locals, acc_locals = [], []  # Store training loss & accuracy
    
    # Select clients randomly for this round
    selected_clients = np.random.choice(range(args.num_users), max(1, int(args.frac * args.num_users)), replace=False)

    for client_id in selected_clients:
        local_model = SNN().to(args.device)
        local_model.load_state_dict(global_weights)  # Load global model weights

        # Perform local training on client's dataset
        w, train_loss, train_acc = LocalUpdate(client_id, local_model, dataset_train, dataset_test, 
                                               dict_users_train[client_id], dict_users_test[client_id], args).train()

        w_locals.append(w)  # Store updated weights
        loss_locals.append(train_loss)
        acc_locals.append(train_acc)

    # Aggregate updates using FedAvg
    global_weights = FedAvg(w_locals)
    model.load_state_dict(global_weights)  # Update global model

    # Global Evaluation After Aggregation
    test_loss, test_acc = test(model, test_dataset, classer, error, args)  # Use the full test dataset

    # Store values
    train_loss_collect.append(sum(loss_locals) / len(loss_locals))  # Avg train loss
    train_acc_collect.append(sum(acc_locals) / len(acc_locals))      # Avg train acc
    test_loss_collect.append(test_loss)  # Store test loss
    test_acc_collect.append(test_acc)    # Store test accuracy

    print(f'\033[F\rEpoch [{epoch+1}/{args.epochs}] '
          f'Train Loss: {train_loss_collect[-1]:.4f}, Train Acc: {train_acc_collect[-1]:.2f}% '
          f'| Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')




import pandas as pd
from pandas import DataFrame

# Ensure all lists are the same length
min_length = min(len(train_acc_collect), len(test_acc_collect))

round_process = list(range(1, min_length + 1))  # Ensure it matches the shortest list
acc_train_collect = train_acc_collect[:min_length]  # Trim excess values
acc_test_collect = test_acc_collect[:min_length]  # Trim excess values

# Create DataFrame
df = DataFrame({
    'Round': round_process, 
    'Train Accuracy (%)': acc_train_collect,
    'Test Accuracy (%)': acc_test_collect
})

# Save to Excel
file_name = "FL_Results.xlsx"    
df.to_excel(file_name, sheet_name="Federated_Learning_Results", index=False)

print(f"Training results saved as {file_name}")

