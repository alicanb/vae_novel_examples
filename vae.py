import os
from numbers import Number

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from tqdm.auto import tqdm

import mlp


class VAE(nn.Module):
    def __init__(self, im_shape=None, dim_z=10, encoder=256, decoder=None, beta=1., device='cpu'):
        super(VAE, self).__init__()
        self.config = {'dim_z': dim_z, 'beta': beta}

        if isinstance(encoder, Number):
            encoder = [encoder]

        if isinstance(encoder, nn.Module):
            self.enc = encoder
            self.config['enc'] = []
        elif isinstance(encoder, list):
            self.enc = mlp.Encoder(im_shape, encoder, dim_z=dim_z)
            self.config['enc'] = encoder

        if decoder is None:
            if isinstance(encoder, list):
                decoder = encoder[::-1]
            else:
                raise ValueError('Must provide decoder with custom Encoders.')
        elif isinstance(decoder, Number):
            decoder = [decoder]

        if isinstance(decoder, nn.Module):
            self.dec = decoder
            self.config['dec'] = []
        elif isinstance(decoder, list):
            self.dec = mlp.Decoder(im_shape, decoder, dim_z=dim_z)
            self.config['dec'] = encoder

        self.device = device
        self.to(device)

        self.loss_fn = ELBOLoss(beta=beta)

    @property
    def dim_z(self):
        return self.config['dim_z']

    def encode(self, x):
        return self.enc(x)

    @staticmethod
    def sample_from(mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.sample_from(mu, log_std)
        xhat = self.decode(z)
        return xhat, mu, log_std

    def run_epoch(self, loader, is_train=False, optimizer=None):
        self.train(is_train)
        total_loss = 0
        total_rate = 0
        total_distortion = 0
        with torch.set_grad_enabled(is_train):
            for batch_idx, (data, _) in enumerate(loader):
                data = data.to(self.device)
                if isinstance(optimizer, optim.Optimizer):
                    optimizer.zero_grad()
                recon_batch, mu, log_std = self(data)
                loss, rate, dist = self.loss_fn(recon_batch, data, mu, log_std)
                if is_train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                total_rate += rate.item()
                total_distortion += dist.item()
        total_loss /= len(loader.dataset)
        total_rate /= len(loader.dataset)
        total_distortion /= len(loader.dataset)
        return total_loss, total_rate, total_distortion

    def reconstruct(self, x, use_mean=False):
        mu, log_var = self.enc(x)
        z = mu if use_mean else self.sample_from(mu, log_var)
        return self.decode(z)

    def fit(self, train_loader, test_loader=None, num_epochs=1, optimizer=None):
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)

        prog_bar = tqdm(range(1, num_epochs + 1))
        for epoch in prog_bar:
            train_loss, train_rate, train_distortion = self.run_epoch(train_loader, is_train=True, optimizer=optimizer)
            metrics = {'Train ELBO': -train_loss, 'Train R': train_rate, 'Train D': train_distortion}
            if test_loader is not None:
                test_loss, test_rate, test_distortion = self.run_epoch(test_loader, is_train=False)
                metrics.update({'Test ELBO': -test_loss, 'Test R': test_rate, 'Test D': test_distortion})
            prog_bar.set_postfix(ordered_dict=metrics)

    def save(self, filename=None, optimizer=None, model_dir='./models'):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        if filename is None:
            filename = '_'.join([self.enc.filename, self.dec.filename]) + '.tar'
        param_dict = {'enc_state_dict': self.enc.state_dict(),
                      'dec_state_dict': self.dec.state_dict(),
                      }
        param_dict.update(self.config)
        if optimizer is not None:
            param_dict['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(param_dict, os.path.join(model_dir, filename))

    @staticmethod
    def load(filename, model=None, enc_file=None, dec_file=None, optimizer=None):
        checkpoint = torch.load(filename)
        if model is None:
            model = VAE(dim_z=checkpoint['dim_z'], encoder=checkpoint['enc'], decoder=checkpoint['dec'])
        if enc_file is not None:
            model.enc = torch.load(enc_file)
        else:
            model.enc.load_state_dict(checkpoint['enc_state_dict'])
        if dec_file is not None:
            model.dec = torch.load(enc_file)
        else:
            model.dec.load_state_dict(checkpoint['dec_state_dict'])

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except KeyError:
            pass
        return model, optimizer


class ELBOLoss:
    def __init__(self, beta=1., reduction='sum'):
        self.beta = beta
        reduction = reduction.lower()
        if reduction in ['none', 'sum', 'mean']:
            self.reduction = reduction

    @staticmethod
    def rate(mu, log_std, reduction='sum'):
        rate = -0.5 * (1 + 2*log_std - mu.pow(2) - (2*log_std).exp())
        if reduction == 'sum':
            rate = torch.sum(rate)
        elif reduction == 'mean':
            rate = torch.mean(rate)
        elif not reduction == 'none':
            raise ValueError("reduction must be ['none', 'sum', 'mean]")
        return rate

    @staticmethod
    def distortion(recon_x, x, reduction='sum'):
        return F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction=reduction)

    def __call__(self, recon_x, x, mu, log_std):
        rate = self.rate(mu,log_std,reduction=self.reduction)
        distortion = self.distortion(recon_x, x, reduction=self.reduction)
        return distortion + self.beta * rate, rate, distortion

