import numpy as np
import torch
import torch.distributions as dists

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from vae import VAE


def show_grid(images, labels=None, num_rows=None, num_cols=None, axs=None,
              random=False, figsize=None, **kwargs):
    if axs is None:
        if num_rows is None and num_cols is None:
            num_rows = int(np.floor(np.sqrt(images.shape[0])))
            num_cols = int(np.ceil(np.sqrt(images.shape[0])))
        elif num_rows is not None:
            num_cols = int(np.ceil(images.shape[0] / num_rows))
        else:
            num_rows = int(np.ceil(images.shape[0] / num_cols))
        _, axs = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=figsize)

    num_images = images.shape[0]
    num_axes = min(axs.size, num_images)

    ind = np.random.choice(num_images, num_axes, replace=False) if random else np.arange(num_axes)
    for i, ax in zip(ind, axs.flatten()):
        im = ax.imshow(images[i], **kwargs)
        if labels is not None:
            ax.set_title(labels[i])
        plt.colorbar(im, ax=ax)
        ax.axis('off')

    return axs


def show_prior(model, num_samples=100):
    with torch.no_grad():
        z = torch.randn(num_samples, model.num_z, device=model.device)
        images = model.decode(z)
    show_grid(images, cmap='gray', vmin=0, vmax=1)
