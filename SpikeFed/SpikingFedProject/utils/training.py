

import torch
def train(model, data, opt, err, pred, args):
  model.train()
  total_samples = 0
  loss_sum = 0
  total_correct_samples = 0
  for i, (images, labels) in enumerate(data):

    # Run Model #
    images = images.to(args.device)
    images = images.to(torch.float32)
    labels = labels.to(args.device)
    labels = labels.to(torch.int64)    
    outputs = model(images)

    # Loss #
    loss = err(outputs, labels)
    loss.backward()
    opt.step()
    opt.zero_grad()


    # Stats #
    total_samples += images.shape[0]
    loss_sum += loss.cpu().data.item() * outputs.shape[0]
    correct_samples = torch.sum(pred(outputs)==labels).cpu().data.item()
    total_correct_samples += correct_samples

    # Print Stats #
    acc = total_correct_samples / total_samples

    print(f'\r\tBatch [{i+1}/{len(data)}] Training: {acc:.2%}',end="")

  return acc



import torch  # Ensure this import is included

CHANNEL_WARNING_PRINTED_TEST = False  # Declare this at the top of the script

def test(model, data, pred, error, args):
    global CHANNEL_WARNING_PRINTED_TEST  # Declare global usage

    model.eval()
    total_samples = 0
    total_correct_samples = 0
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data):
            images = torch.tensor(images, dtype=torch.float32).to(args.device)
            labels = torch.tensor(labels, dtype=torch.int64).to(args.device)

            # if images.dim() == 4:  
            #     images = images.unsqueeze(1)  

            # if images.shape[1] == 1:  
            #     if not CHANNEL_WARNING_PRINTED_TEST:
            #         print(f"⚠️ Warning: Expected input channel=2, but got 1. Adjusting... (This warning will only appear once.)")
            #         CHANNEL_WARNING_PRINTED_TEST = True  
            #     images = images.repeat(1, 2, 1, 1, 1)  

            outputs = model(images)

            total_loss += error(outputs, labels).cpu().data.item()
            total_samples += images.shape[0]
            correct_samples = torch.sum(pred(outputs) == labels).cpu().data.item()
            total_correct_samples += correct_samples

            acc = total_correct_samples / total_samples
            print(f'\r\tBatch [{i+1}/{len(data)}] Validation: {acc:.2%}', end="")

    return total_loss / i, acc





###############################################################################
#local client training
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class LocalUpdate(object):
    def __init__(self, idx, lr, device, dataset_train, dataset_test, idxs, idxs_test):
        """Initializes the local training process for each client."""
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=False)

    def train(self, net):
        """Performs local training on the client's data."""
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        epoch_loss = []

        for _ in range(self.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                fx = net(images)
                loss = self.loss_func(fx, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def evaluate(self, net):
        """Evaluates the local model."""
        net.eval()
        with torch.no_grad():
            batch_loss = []
            for images, labels in self.ldr_test:
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)
                loss = self.loss_func(fx, labels)
                batch_loss.append(loss.item())

        return sum(batch_loss) / len(batch_loss)
