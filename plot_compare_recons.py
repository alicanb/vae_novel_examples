import argparse
import glob

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from plot_utils import compare_reconstructions
from vae import VAE
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Plot Reconstruction')
parser.add_argument('--model', type=str, default='./models/holdout9_z50*',
                    help='input batch size for training (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--spt', type=int, default=10,
                    help='samples per target')
parser.add_argument('--holdout-digit', type=int, default=9,
                    help='holdout digit')
parser.add_argument('--num-neurons', nargs='+', type=int)
parser.add_argument('--index', type=int, default=1)

args = parser.parse_args()
torch.manual_seed(args.seed)
device = "cpu"


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def count_params(module):
    return sum([p.numel() for p in module.parameters() if p.requires_grad])


if __name__ == "__main__":
    test_targets = [args.holdout_digit]
    train_targets = list(range(10))
    train_targets.remove(args.holdout_digit)
    train_data = datasets.HoldoutMNIST('.', targets=train_targets, train=True,
                                       download=True, transform=transforms.ToTensor())
    test_data = datasets.HoldoutMNIST('.', targets=test_targets, train=False,
                                      download=True, transform=transforms.ToTensor())
    train_imgs = next(DataLoader(train_data, batch_size=len(train_data), shuffle=False).__iter__())[0]
    test_imgs = next(DataLoader(test_data, batch_size=args.index+1, shuffle=False).__iter__())[0][args.index]

    num_params = []
    num_layers = []
    recons_images = []
    weighted_averages = []
    for path in glob.glob(args.model):
        model, _ = VAE.load(path)
        num_params.append(count_params(model.dec))
        num_layers.append(model.dec.num_layers)
        with torch.no_grad():
            recons_images.append(model.reconstruct(test_imgs, use_mean=True))
            weighted_averages.append(model.optimal_reconstruct(test_imgs, train_data=train_imgs))

    fig = compare_reconstructions(num_params, num_layers, test_imgs, recons_images, weighted_averages)
    plt.show()