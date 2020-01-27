import argparse
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from plot_utils import overlay_jointgrid
from vae import VAE, ELBOLoss

parser = argparse.ArgumentParser(description='Plot BCE between input, output, and weighted average')
parser.add_argument('--model-dir', type=str, default='./models/',
                    help='model path')
parser.add_argument('--arch1', nargs='+', default=('400',), help='encoder arch for model1')
parser.add_argument('--arch2', nargs='+', default=('400', '200', '100'), help='encoder arch for model2')
parser.add_argument('--holdout-digit', type=int, default=9,
                    help='holdout digit')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--dim-z', type=int, default=50,
                    help='latent dimension (default:50)')

args = parser.parse_args()
device = "cpu"


def model_path(holdout, arch):
    return os.path.join(args.model_dir, '{}{}_z{}_arch{}.tar'.format('holdout' if holdout else 'full',
                                                                     args.holdout_digit,
                                                                     args.dim_z,
                                                                     '_'.join(arch)
                                                                     ))


def batch_bce(xhat, x):
    return ELBOLoss.distortion(xhat.view(xhat.shape[0], -1),
                               x.view(x.shape[0], -1),
                               'none').sum(-1)


def get_bce(holdout, arch):
    test_targets = [args.holdout_digit, ]
    train_targets = list(range(10))
    if holdout:
        train_targets.remove(args.holdout_digit)
    train_data = datasets.HoldoutMNIST('.', targets=train_targets, train=True,
                                       download=True, transform=transforms.ToTensor())
    test_data = datasets.HoldoutMNIST('.', targets=test_targets, train=False,
                                      download=True, transform=transforms.ToTensor())
    train_imgs = next(DataLoader(train_data, batch_size=len(train_data), shuffle=False).__iter__())[0]
    test_imgs = next(DataLoader(test_data, batch_size=len(test_data), shuffle=False).__iter__())[0]

    model, _ = VAE.load(model_path(holdout, arch))
    model.eval()
    with torch.no_grad():
        # TODO: make these work with dataloaders so we don't need the for loop
        bce_in_out = []
        bce_wa_out = []
        for i in range(0, test_imgs.shape[0], args.batch_size):
            rec_imgs = model.reconstruct(test_imgs[i:i + args.batch_size], use_mean=True)
            w_avgs = model.optimal_reconstruct(test_imgs[i:i + args.batch_size], train_data=train_imgs)
            bce_in_out.append(batch_bce(rec_imgs, test_imgs[i:i + args.batch_size]))
            bce_wa_out.append(batch_bce(rec_imgs, w_avgs))
        bce_in_out = torch.cat(bce_in_out).numpy()
        bce_wa_out = torch.cat(bce_wa_out).numpy()
    return bce_wa_out, bce_in_out


if __name__ == "__main__":
    model1_holdout = get_bce(True, args.arch1)
    model1_full = get_bce(False, args.arch1)
    model2_holdout = get_bce(True, args.arch2)
    model2_full = get_bce(False, args.arch2)
    top_title = r'Decoder (${}\rightarrow {} \rightarrow 784$)'.format(args.dim_z,
                                                                       r' \rightarrow '.join(args.arch1[::-1]))
    bottom_title = r'Decoder (${}\rightarrow {} \rightarrow 784$)'.format(args.dim_z,
                                                                          r' \rightarrow '.join(args.arch2[::-1]))
    overlay_jointgrid(model1_full, model1_holdout, model2_full, model2_holdout,
                      top_title=top_title, bottom_title=bottom_title,
                      x_label='$BCE(\mu,\hat{x})$', y_label='$BCE(x,\hat{x})$', x_lim=[0, 250], y_lim=[0, 250],
                      label='Trained with 9s', label2='Trained without 9s', label3='Trained with 9s',
                      label4='Trained without 9s', top_zorders=(1, 0), bottom_zorders=(1, 0))
    plt.show()
