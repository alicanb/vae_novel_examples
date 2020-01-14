import torch
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from numbers import Number
import mlp
from tqdm.auto import tqdm


class VAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, beta=1., device='cpu'):
        super(VAE, self).__init__()
        if encoder is None:
            encoder = [256]
        elif isinstance(encoder, Number):
            encoder = [encoder]

        if isinstance(encoder, nn.Module):
            self.enc = encoder
        elif isinstance(encoder, list):
            self.enc = mlp.Encoder(encoder)

        if decoder is None:
            if isinstance(encoder, list):
                decoder = encoder[::-1]
            else:
                raise ValueError('Must provide decoder with custom Encoders.')
        elif isinstance(decoder, Number):
            decoder = [decoder]

        if isinstance(decoder, nn.Module):
            self.dec = decoder
        elif isinstance(decoder, list):
            self.enc = mlp.Decoder(decoder)

        self.device = device
        self.to(device)
        #self.enc.to(device)
        #self.dec.to(device)

        self.loss_fn = ELBOLoss(beta=beta)

    def encode(self, x):
        return self.enc(x)

    @staticmethod
    def sample_from(mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.sample_from(mu, log_var)
        return self.decode(z), mu, log_var

    def run_epoch(self, loader, is_train=False):
        self.train(is_train)
        total_loss = 0
        total_rate = 0
        total_distortion = 0
        with torch.set_grad_enabled(is_train):
            for batch_idx, (data, _) in enumerate(loader):
                data = data.to(self.device)
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
        mu, log_var = self.enc(x.view(-1, 784))
        z = mu if use_mean else self.sample_from(mu, log_var)
        return self.decode(z)

    def fit(self, train_loader, test_loader=None, num_epochs=1):
        for epoch in range(1, num_epochs + 1):
            self.run_epoch(train_loader, is_train=True)
            if test_loader is not None:
                self.run_epoch(test_loader, is_train=False)

    def save(self, model_dir):
        torch.save(self, )


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

