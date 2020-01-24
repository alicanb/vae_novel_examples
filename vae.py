import math
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
            self.config['dec'] = decoder

        self.device = device
        self.to(device)

        self.loss_fn = ELBOLoss(beta=beta)

    @property
    def dim_z(self):
        return self.config['dim_z']

    def encode(self, x, batch_size=None):
        if batch_size is None:
            return self.enc(x)
        else:
            mu_list = []
            logstd_list = []
            for i in range(0, x.shape[0], batch_size):
                mu, log_std = self.enc(x[i:i + batch_size])
                mu_list.append(mu)
                logstd_list.append(log_std)
            return torch.cat(mu_list, dim=0), torch.cat(logstd_list, dim=0)

    @staticmethod
    def sample_from(mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, batch_size=None):
        if batch_size is None:
            return self.dec(z)
        else:
            return torch.cat([self.dec(z[i:i + batch_size]) for i in range(0, z.shape[0], batch_size)], dim=0)

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

    def reconstruct(self, x, use_mean=False, batch_size=64):
        mu, log_std = self.encode(x, batch_size)
        z = mu if use_mean else self.sample_from(mu, log_std)
        return self.decode(z, batch_size)

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

    def log_posterior(self, z: torch.Tensor, train_data=None, train_mu=None, train_log_std=None, batch_size=100):
        with torch.no_grad():
            batch_dims = z.shape[:-1]
            z = z.reshape(-1, z.shape[-1])
            if train_mu is None and train_log_std is None:
                train_mu, train_log_std = self.encode(train_data)
            train_mu = train_mu.unsqueeze(0)  # (1 x K x D)
            train_log_std = train_log_std.unsqueeze(0)  # (1 x K x D)
            z = z.unsqueeze(1)  # (N x 1 x D)
            weights = [normal_logprob(z[:, i:i + batch_size], train_mu, train_log_std).sum(-1) for i in
                       range(0, z.shape[1], batch_size)]
            weights = torch.cat(weights, -1)  # N x K
            weights = weights.reshape(batch_dims + weights.shape[-1:])
        return weights

    def agg_posterior(self, z: torch.Tensor, train_data=None, train_mu=None, train_log_std=None, batch_size=100):
        log_probs = self.log_posterior(z=z, train_data=train_data, train_mu=train_mu,
                                       train_log_std=train_log_std, batch_size=batch_size)
        with torch.no_grad():
            log_probs = log_probs.logsumexp(dim=-1, keepdim=False) - math.log(log_probs.shape[-1])  # logmeanexp
            log_probs.exp_()
        return log_probs

    def optimal_weights(self, z: torch.Tensor, train_data=None, train_mu=None, train_log_std=None, batch_size=100):
        weights = self.log_posterior(z=z, train_data=train_data, train_mu=train_mu,
                                     train_log_std=train_log_std, batch_size=batch_size)
        with torch.no_grad():
            # normalize
            weights = weights - weights.logsumexp(dim=-1, keepdim=True)
            weights.exp_()
        return weights

    def optimal_reconstruct(self, test_data=None, test_mu=None, train_data=None,
                            train_mu=None, train_log_std=None, batch_size=100):
        with torch.no_grad():
            if test_mu is None:
                test_mu, _ = self.encode(test_data, batch_size=batch_size)
            weights = self.optimal_weights(test_mu, train_data=train_data, train_mu=train_mu,
                                           train_log_std=train_log_std, batch_size=batch_size)
            weighted_average = torch.einsum('...i, i...jk->...jk', weights, train_data)
        return weighted_average


class ELBOLoss:
    def __init__(self, beta=1., reduction='sum'):
        self.beta = beta
        reduction = reduction.lower()
        if reduction in ['none', 'sum', 'mean']:
            self.reduction = reduction

    @staticmethod
    def rate(mu, log_std, reduction='sum'):
        rate = -0.5 * (1 + 2 * log_std - mu ** 2 - (2 * log_std).exp())
        if reduction == 'sum':
            rate = torch.sum(rate)
        elif reduction == 'mean':
            rate = torch.mean(rate)
        elif not reduction == 'none':
            raise ValueError("reduction must be ['none', 'sum', 'mean]")
        return rate

    @staticmethod
    def distortion(recon_x, x, reduction='sum'):
        eps = torch.finfo(recon_x.dtype).eps
        recon_x = recon_x.clamp(min=eps, max=1 - eps)
        return F.binary_cross_entropy(recon_x, x, reduction=reduction)

    def __call__(self, recon_x, x, mu, log_std):
        rate = self.rate(mu, log_std, reduction=self.reduction)
        distortion = self.distortion(recon_x, x, reduction=self.reduction)
        return distortion + self.beta * rate, rate, distortion


def normal_logprob(z, mu, log_std):
    return -0.5 * ((z - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
