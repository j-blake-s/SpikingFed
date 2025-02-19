import torch
import os
from torch.utils.data import DataLoader
from utils.training import train, test


# Args
from utils.argparser import get_args
args = get_args()

# Data

if args.dataset == 'cifar10dvs':
  from utils.cf10data import Cifar10DVS
  print("Loading Cifar10 DVS Dataset...")
  dataset_train, dataset_test = Cifar10DVS("/data/CIFAR10DVS/dataset")
  args.classes = 10
elif args.dataset == 'dvsgesture':
  from utils.gesturedata import DvsGesture
  path = "/data/DvsGesture/t16"
  print("Loading DVS Gesture Dataset...")
  dataset_train = DvsGesture(os.path.join(path, "train.npz"))
  dataset_test = DvsGesture(os.path.join(path, "test.npz"))
  args.classes = 11


# Model
if args.model == "ann":
  from model.ann import ANN
  print("Loading ANN model...")
  model = ANN(args).to(args.device)
  error = torch.nn.CrossEntropyLoss()
  classer = lambda x: torch.argmax(x, dim=-1)

elif args.model == "snn":
  from model.snn import SNN
  print("Loading SNN model...")
  model = SNN(args).to(args.device)
  error = slayer.loss.SpikeRate(true_rate=0.5, false_rate=0.05, reduction='mean').to(args.device)
  classer = slayer.classifier.Rate.predict
print(f'Parameters: {model.params():,}')




print(f'Found {len(dataset_train):,} training samples...')
print(f'Found {len(dataset_test):,} testing samples...')

data_loader = torch.utils.data.DataLoader(
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
    pin_memory=True)


# Learning Tools
import lava.lib.dl.slayer as slayer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



# # Run
# torch.autograd.set_detect_anomaly(True)
for epoch in range(args.epochs):
  print(" "*50,end="\r")
  print(f'Epoch [{epoch+1}/{args.epochs}]')
  train_acc = train(model, data_loader, optimizer, error, classer, args)
  test_acc = test(model, data_loader_test, classer, args)
  print(f'\033[F\rEpoch [{epoch+1}/{args.epochs}] Training: {train_acc:.2%}\tValidation: {test_acc:.2%}              ')