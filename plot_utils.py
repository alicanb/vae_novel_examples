import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch


def show_grid(images, labels=None, num_rows=None, num_cols=None, axs=None,
              random=False, figsize=None, colorbar=False, **kwargs):
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
        if colorbar:
            plt.colorbar(im, ax=ax)
        ax.axis('off')

    return axs


def show_prior(model, num_samples=100):
    with torch.no_grad():
        z = torch.randn(num_samples, model.num_z, device=model.device)
        images = model.decode(z)
    show_grid(images, cmap='gray', vmin=0, vmax=1)


def compare_reconstructions(num_params, num_layers, orig_images, recons_images, weighted_averages, labels=None):
    num_examples = len(recons_images)
    sorted_i = np.argsort(num_params)
    with sns.axes_style("whitegrid"):
        fig = plt.figure(figsize=(20, 5))
        grid = plt.GridSpec(2, 1, hspace=0.07, wspace=0.00)
        g0 = mpl.gridspec.GridSpecFromSubplotSpec(2, num_examples + 1, subplot_spec=grid[0], hspace=0.07, wspace=0.00)
        main_ax = fig.add_subplot(grid[1])
        im = main_ax.plot(num_params, num_layers, 'ko')
        if labels is not None:
            for prm, lyr, lbl in zip(num_params, num_layers, labels):
                plt.text(prm, lyr, lbl, rotation=-25, fontsize=8)
        plt.semilogx()
        plt.grid(axis='x', which='both')
        formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.4))
        main_ax.get_xaxis().set_minor_formatter(formatter)
        main_ax.get_xaxis().set_major_formatter(formatter)
        plt.xlabel('Decoder parameters')
        plt.ylabel('Hidden layers')
        # plt.tight_layout()
        # sns.despine(offset=10, trim=True);
        axs_orig = fig.add_subplot(g0[0:2, 0])
        axs_orig.imshow(orig_images[0].reshape(28, 28))
        axs_orig.set_axis_off()
        plt.title("Input")
        plt.tight_layout()
        axs_wa = [fig.add_subplot(g0[0, i + 1]) for i in range(num_examples)]
        axs_dec = [fig.add_subplot(g0[1, i + 1]) for i in range(num_examples)]
        for i in range(num_examples):
            axs_wa[i].get_xaxis().set_ticks([])
            axs_wa[i].get_yaxis().set_ticks([])
            axs_wa[i].imshow(weighted_averages[sorted_i[i]][0])
            plt.tight_layout()
            axs_dec[i].get_xaxis().set_ticks([])
            axs_dec[i].get_yaxis().set_ticks([])
            axs_dec[i].imshow(recons_images[sorted_i[i]][0])
            plt.tight_layout()
            con = ConnectionPatch(xyA=(14, 27), xyB=(num_params[sorted_i[i]], num_layers[sorted_i[i]]), coordsA="data",
                                  coordsB="data",
                                  axesA=axs_dec[i], axesB=main_ax, color="red")
            axs_dec[i].add_artist(con)
        fig.text(0.5, 1.00, 'Weighted average (top) and decoder output (bottom)', ha='center')
        # axs_wa[0].set_ylabel('Weighted\naverage')
        plt.tight_layout(h_pad=0.5)
    plt.draw()
    return fig
