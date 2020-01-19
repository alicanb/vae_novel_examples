import argparse

import torch.utils.data
from torchvision import transforms
from tqdm.auto import tqdm

import datasets
from vae import VAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dim-z', type=int, default=50,
                    help='latent dimension (default:50)')
parser.add_argument('--holdout', action='store_true', default=False,
                    help='holdout 9 from training if True')
parser.add_argument('--beta', type=float, default=1.,
                    help='coefficient of rate term in loss')
parser.add_argument('--holdout-digit', type=int, default=9,
                    help='holdout digit')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = "cuda" if args.cuda else "cpu"

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if __name__ == "__main__":
    test_targets = [args.holdout_digit]
    train_targets = list(range(10))
    if args.holdout:
        train_targets.remove(args.holdout_digit)
    train_loader = torch.utils.data.DataLoader(
        datasets.HoldoutMNIST('.', targets=train_targets, train=True,
                              download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.HoldoutMNIST('.', targets=test_targets, train=False,
                              download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True)
    for enc in tqdm([[512, 512, 512]]):
        vae = VAE(im_shape=[1, 28, 28], dim_z=args.dim_z, encoder=enc, device=device, beta=args.beta)
        vae.fit(train_loader, test_loader, num_epochs=args.epochs)
        filename = '{prefix}{digit}_z{dim_z}_arch{arch}.tar'.format(prefix='holdout' if args.holdout else 'full',
                                                                    digit=args.holdout_digit,
                                                                    dim_z=args.dim_z,
                                                                    arch='_'.join([str(a) for a in enc]))
        vae.save(filename=filename)
