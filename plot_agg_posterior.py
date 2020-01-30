import argparse

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from plot_utils import scatter_encodings
from vae import VAE

parser = argparse.ArgumentParser(description='Plot Aggregate Posterior')
parser.add_argument('--model', type=str, default='./models/mini9_z2_arch400_200_100.tar',
                    help='input batch size for training (default: 128)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--spt', type=int, default=10,
                    help='samples per target')
parser.add_argument('--extent', type=int, nargs='+', default=[-5, 5, -5, 5],
                    help='bounds of latent space')
parser.add_argument('--cmap-qz', type=str, default='Oranges_r', help='colormap for aggregate posterior')
parser.add_argument('--cmap-w', type=str, default='Oranges_r', help='colormap for weights')
parser.add_argument('--num-points', type=int, default=600, help='number of points evaluated points per latent dim')
parser.add_argument('--save-fig', action='store_true', default=False, help='save the figure')
parser.add_argument('--save-name', type=str, default='agg_posterior.png', help='name for saved image')

args = parser.parse_args()
torch.manual_seed(args.seed)
device = "cpu"

if __name__ == "__main__":
    model, _ = VAE.load(args.model)
    train_data = datasets.MiniMNIST('.', train=True, seed=args.seed, transform=transforms.ToTensor())
    train_imgs = next(DataLoader(train_data, batch_size=len(train_data), shuffle=False).__iter__())[0]

    z = np.meshgrid(
        np.linspace(args.extent[0], args.extent[1], args.num_points),
        np.linspace(args.extent[2], args.extent[3], args.num_points))
    z = torch.tensor(np.stack(z, -1)).float()
    with torch.no_grad():
        mus, log_stds = model.encode(train_imgs)
    qz = model.agg_posterior(z, train_mu=mus, train_log_std=log_stds)
    w = model.optimal_weights(z, train_mu=mus, train_log_std=log_stds)
    w_max = torch.max(w, dim=-1)[0]

    with sns.axes_style("white"):
        fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
        im2 = axs[0].imshow(qz, extent=args.extent, cmap=args.cmap_qz, vmin=0, vmax=1)
        axs[0].set_xlim(args.extent[0], args.extent[1])
        axs[0].set_ylim(args.extent[2], args.extent[3])
        axs[0].set_xlabel('$z_1$')
        axs[0].set_ylabel('$z_2$')
        fig.colorbar(im2, orientation='horizontal', fraction=0.046, pad=0.1, ax=axs[0])
        plt.tight_layout()

        largest_w = ndimage.grey_erosion(w_max, size=(3, 3))
        im = axs[1].imshow(np.flipud(largest_w), extent=args.extent, cmap=args.cmap_w, interpolation='nearest')
        scatter_encodings(mus, log_stds.exp(), ax=axs[1], alpha=1)
        axs[1].set_xlabel('$z_1$')
        axs[1].set_ylabel('$z_2$')
        axs[1].set_xlim(args.extent[0], args.extent[1])
        axs[1].set_ylim(args.extent[2], args.extent[3])
        fig.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.1, ax=axs[1])
        plt.tight_layout()

    axs[0].set_title("$q_{\phi}(z)$")
    axs[1].set_title("$w_{max}(z)$")
    plt.show()
    if args.save_fig:
        fig.savefig(args.save_name)
