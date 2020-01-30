import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from plot_utils import show_grid
from vae import VAE

parser = argparse.ArgumentParser(description='Plot weight perplexity histogram')
parser.add_argument('--enc-arch', type=str, nargs='+', help='encoder architecture', required=True)
parser.add_argument('--holdout-digit', type=int, default=9,
                    help='holdout digit')
parser.add_argument('--holdout', action='store_true', default=False,
                    help='holdout HOLDOUT_DIGIT from training if True')
parser.add_argument('--model-dir', type=str, default='./models/',
                    help='model path')
parser.add_argument('--dim-z', type=int, default=50,
                    help='latent dimension (default:50)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--index', type=int, nargs='+', default=(0, 1, 2),
                    help='index')
parser.add_argument('--num-nearest', type=int, default=3, help='number of training samples with largest weight')
parser.add_argument('--save-fig', action='store_true', default=False, help='save the figure')
parser.add_argument('--save-name', type=str, default='agg_posterior.png', help='name for saved image')

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
    test_imgs = torch.stack([test_data[ind][0] for ind in args.index])

    train_targets = list(range(10))
    if args.holdout:
        train_targets.remove(args.holdout_digit)
    train_data = datasets.HoldoutMNIST('.', targets=train_targets, train=True,
                                       download=True, transform=transforms.ToTensor())
    train_imgs = next(DataLoader(train_data, batch_size=len(train_data), shuffle=False).__iter__())[0]

    model, _ = VAE.load(model_path(args.holdout, args.enc_arch))
    model.eval()

    with torch.no_grad():
        z = model.encode(test_imgs)[0]
        weights = model.optimal_weights(z, train_data=train_imgs)
        wavgs = model.optimal_reconstruct(weights=weights, train_data=train_imgs)
        recons = model.decode(z)
        max_w, max_w_inds = weights.topk(k=args.num_nearest, dim=-1)
        wmax_imgs = train_imgs[max_w_inds, :, :]
        imgs = torch.stack([test_imgs, recons, wavgs, *torch.unbind(wmax_imgs, dim=1)], dim=1)
        imgs = imgs.reshape(-1, *imgs.shape[-3:]).squeeze()
    fig, axs = plt.subplots(nrows=len(args.index), ncols=3 + args.num_nearest, figsize=(2 * (args.num_nearest + 3),
                                                                                        2 * len(args.index))
                            )
    axs = show_grid(imgs, axs=axs, cmap='gray', vmin=0, vmax=1)
    axs[0, 0].set_title('Input')
    axs[0, 1].set_title('Output')
    axs[0, 2].set_title('Weighted\nAverage')
    for (i, j), w in np.ndenumerate(np.round(max_w, 2)):
        axs[i, j + 3].set_title(r'$w:{:.2}$'.format(w))
    plt.show()
    if args.save_fig:
        fig.savefig(args.save_name)
