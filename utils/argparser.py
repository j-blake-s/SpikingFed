
import argparse


def get_args():
    parser = argparse.ArgumentParser('Cifar10DVS')

    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--model', type=str, default='snn')
    parser.add_argument('--dataset', type=str, default='cifar10dvs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=200)

    arguments = parser.parse_args()
    return arguments
