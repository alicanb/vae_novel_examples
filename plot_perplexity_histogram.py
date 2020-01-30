import argparse
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from plot_utils import entropy_histogram
from vae import VAE

parser = argparse.ArgumentParser(description='Plot weight perplexity histogram')
parser.add_argument('--enc-arch', type=str, nargs='+', help='encoder architecture')
parser.add_argument('--holdout-digit', type=int, default=9,
                    help='holdout digit')
parser.add_argument('--index', type=int, nargs='+', default=(0,))
parser.add_argument('--holdout', action='store_true', default=False,
                    help='holdout HOLDOUT_DIGIT from training if True')
parser.add_argument('--model-dir', type=str, default='./models/',
                    help='model path')
parser.add_argument('--dim-z', type=int, default=50,
                    help='latent dimension (default:50)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--save-fig', action='store_true', default=False, help='save the figure')
parser.add_argument('--save-name', type=str, default='weight_perplexity.png', help='name for saved image')
args = parser.parse_args()
device = "cpu"


def model_path(holdout, arch):
    return os.path.join(args.model_dir, '{}{}_z{}_arch{}.tar'.format('holdout' if holdout else 'full',
                                                                     args.holdout_digit,
                                                                     args.dim_z,
                                                                     '_'.join(arch)
                                                                     ))


if __name__ == "__main__":
    test_targets = [args.holdout_digit]
    test_data = datasets.HoldoutMNIST('.', targets=test_targets, train=False,
                                      download=True, transform=transforms.ToTensor())
    test_imgs = next(DataLoader(test_data, batch_size=len(test_data), shuffle=False).__iter__())[0]

    train_targets = list(range(10))
    if args.holdout:
        train_targets.remove(args.holdout_digit)
    train_data = datasets.HoldoutMNIST('.', targets=train_targets, train=True,
                                       download=True, transform=transforms.ToTensor())
    train_imgs = next(DataLoader(train_data, batch_size=len(train_data), shuffle=False).__iter__())[0]

    model, _ = VAE.load(model_path(args.holdout, args.enc_arch))
    model.eval()

    weights = []
    with torch.no_grad():
        for i in range(0, test_imgs.shape[0], args.batch_size):
            z = model.encode(test_imgs[i:i + args.batch_size])[0]
            weights.append(model.optimal_weights(z, train_data=train_imgs))
        weights = torch.cat(weights)

    fig, ax = plt.subplots(figsize=(3, 3))
    _ = entropy_histogram(weights, perplexity=True, title="Perplexity Histogram", ax=ax)
    plt.show()
    if args.save_fig:
        fig.savefig(args.save_name)
