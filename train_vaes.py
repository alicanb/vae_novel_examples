import argparse
from itertools import combinations_with_replacement

import torch.utils.data
from torch.utils.data import DataLoader
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
parser.add_argument('--mini', action='store_true', default=False,
                    help='use MiniMNIST if True')
parser.add_argument('--spt', type=int, default=10,
                    help='samples per target')
parser.add_argument('--beta', type=float, default=1.,
                    help='coefficient of rate term in loss')
parser.add_argument('--holdout-digit', type=int, default=9,
                    help='holdout digit')
parser.add_argument('--num-neurons', nargs='+', type=int)
parser.add_argument('--enc-arch', nargs='+', type=int)
parser.add_argument('--sweep', action='store_true', default=False)
parser.add_argument('--num-layers', type=int, nargs='+')

args = parser.parse_args()
if args.mini and args.holdout:
    raise ValueError("mini and holdout can't be both true")
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = "cuda" if args.cuda else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if __name__ == "__main__":
    test_targets = [args.holdout_digit]
    train_targets = list(range(10))
    if args.mini:
        train_data = datasets.MiniMNIST('.', samples_per_target=args.spt, train=True,
                                        download=True, transform=transforms.ToTensor(),
                                        random=True, seed=args.seed)
        test_data = datasets.MiniMNIST('.', samples_per_target=args.spt, train=False,
                                       download=True, transform=transforms.ToTensor(),
                                       random=True, seed=args.seed)
    else:
        if args.holdout:
            train_targets.remove(args.holdout_digit)
        train_data = datasets.HoldoutMNIST('.', targets=train_targets, train=True,
                                           download=True, transform=transforms.ToTensor())
        test_data = datasets.HoldoutMNIST('.', targets=test_targets, train=False,
                                          download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    arcs = []
    if args.sweep:
        for num_layer in args.num_layers:
            arcs.extend([list(a[::-1]) for a in combinations_with_replacement(args.num_neurons, r=num_layer)])
    else:
        arcs.append(args.enc_arch)
    for enc in tqdm(arcs):
        vae = VAE(im_shape=[1, 28, 28], dim_z=args.dim_z, encoder=enc, device=device, beta=args.beta)
        vae.fit(train_loader, test_loader, num_epochs=args.epochs)
        if args.holdout:
            prefix = 'holdout'
        elif args.mini:
            prefix = 'mini'
        else:
            prefix = 'full'
        filename = '{prefix}{digit}_z{dim_z}_arch{arch}.tar'.format(prefix=prefix,
                                                                    digit=args.holdout_digit,
                                                                    dim_z=args.dim_z,
                                                                    arch='_'.join([str(a) for a in enc]))
        vae.save(filename=filename)
