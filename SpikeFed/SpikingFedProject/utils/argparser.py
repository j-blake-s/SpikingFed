
import argparse


def get_args():
    parser = argparse.ArgumentParser('Cifar10DVS')
    ########################
    parser.add_argument("--num_users", type=int, default=5, help="Number of clients")
    parser.add_argument("--frac", type=float, default=1, help="Fraction of clients participating per round")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local training epochs per client per round")
    #######################
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--data_path', type=str, default='/data/Hasti/DvsGesture/t16')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=300)

    arguments = parser.parse_args()
    return arguments
