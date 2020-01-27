import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from seaborn import utils as snsutils


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
            con = patches.ConnectionPatch(xyA=(14, 27), xyB=(num_params[sorted_i[i]], num_layers[sorted_i[i]]),
                                          coordsA="data",
                                          coordsB="data",
                                          axesA=axs_dec[i], axesB=main_ax, color="red")
            axs_dec[i].add_artist(con)
        fig.text(0.5, 1.00, 'Weighted average (top) and decoder output (bottom)', ha='center')
        # axs_wa[0].set_ylabel('Weighted\naverage')
        plt.tight_layout(h_pad=0.5)
    plt.draw()
    return fig


def plot_ellipses(mus, stds, ax):
    for mu, std in zip(mus, stds):
        ellipse = patches.Ellipse(
            xy=(mu[0], mu[1]), width=4 * std[0], height=4 * std[1],
            edgecolor='k', fc='None', lw=2)
        ax.add_patch(ellipse)
    return ax


def scatter_encodings(mus, stds, labels=None, ax=None, alpha=0.5):
    if ax is None:
        _, ax_ = plt.subplots()
    else:
        ax_ = ax
    if labels is None:
        ax_.scatter(mus[:, 0], mus[:, 1], alpha=alpha)
        ax_ = plot_ellipses(mus, stds, ax_)
    else:
        for uniq_label in torch.unique(labels):
            mus_u = mus[labels == uniq_label]
            stds_u = stds[labels == uniq_label]
            ax_.scatter(mus_u[:, 0], mus_u[:, 1], alpha=alpha, label='digit=%d'.format(uniq_label))
            ax_ = plot_ellipses(mus_u, stds_u, ax_)
    if ax is None:
        ax_.legend()
    return ax_


def overlay_jointgrid(data, data2, data3, data4,
                      top_title=None, bottom_title=None,
                      x_label=None, y_label=None, x_lim=None, y_lim=None,
                      label=None, label2=None, label3=None, label4=None,
                      top_zorders=(0, 0), bottom_zorders=(0, 0)):
    cmaps = []
    for color in ['r', 'b', 'orange', 'g']:
        color_rgb = mpl.colors.colorConverter.to_rgb(color)
        colors = [snsutils.set_hls_values(color_rgb, l=li)
                  for li in np.linspace(1, 0, 12)]
        cmaps.append(sns.blend_palette(colors, as_cmap=True))

    fig = plt.figure(figsize=(5, 5))
    gs = plt.GridSpec(7, 7)
    ax_joints = [fig.add_subplot(gs[1:4, :-1])]
    ax_joints.append(fig.add_subplot(gs[4:, :-1], sharex=ax_joints[0], sharey=ax_joints[0]))
    ax_marg_x = fig.add_subplot(gs[0, :-1], sharex=ax_joints[0])
    ax_marg_y = fig.add_subplot(gs[1:, -1], sharey=ax_joints[0])
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)
    snsutils.despine(fig)
    snsutils.despine(ax=ax_marg_x, left=True)
    snsutils.despine(ax=ax_marg_y, bottom=True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=.2, wspace=.2)

    # Draw the two density plots and marginals
    for d, cmap, ax_ind, clr, zo in zip([data, data2, data3, data4],
                                        cmaps,
                                        [0, 0, 1, 1],
                                        ['r', 'b', 'orange', 'g'],
                                        top_zorders + bottom_zorders):
        sns.kdeplot(d[0], d[1], cmap=cmap, shade=True, shade_lowest=False, ax=ax_joints[ax_ind], zorder=zo)
        # sns.kdeplot(data[0], data[1],
        #             cmap=cmaps[0], shade=True, shade_lowest=False, ax=ax_joints[0])
        # sns.kdeplot(data2[0], data2[1],
        #             cmap=cmaps[1], shade=True, shade_lowest=False, ax=ax_joints[0])
        # sns.kdeplot(data3[0], data3[1],
        #             cmap=cmaps[2], shade=True, shade_lowest=False, ax=ax_joints[1])
        # sns.kdeplot(data4[0], data4[1],
        #             cmap=cmaps[3], shade=True, shade_lowest=False, ax=ax_joints[1])

        sns.kdeplot(d[0], ax=ax_marg_x, legend=False, color=clr)
        # sns.kdeplot(data[0], ax=ax_marg_x, legend=False, color='r')
        # sns.kdeplot(data2[0], ax=ax_marg_x, legend=False, color='b')
        # sns.kdeplot(data3[0], ax=ax_marg_x, legend=False, color='orange')
        # sns.kdeplot(data4[0], ax=ax_marg_x, legend=False, shade=False, color='g')

        sns.kdeplot(d[1], ax=ax_marg_y, vertical=True, legend=False, color=clr)
        # sns.kdeplot(data[1], ax=ax_marg_y, vertical=True, legend=False, color='r')
        # sns.kdeplot(data2[1], ax=ax_marg_y, vertical=True, legend=False, color='b')
        # sns.kdeplot(data3[1], ax=ax_marg_y, vertical=True, legend=False, color='orange')
        # sns.kdeplot(data4[1], ax=ax_marg_y, vertical=True, legend=False, color='g')

    if x_lim is not None:
        ax_joints[0].set_xlim(x_lim)
    else:
        x_lim = ax_joints[0].xlim()
    if y_lim is not None:
        ax_joints[0].set_ylim(y_lim)
    else:
        y_lim = ax_joints[0].ylim()

    red = sns.color_palette("Reds")[-2]
    blue = sns.color_palette("Blues")[-2]
    green = sns.color_palette("Greens")[-2]
    orange = cmaps[2](0.5)

    # add legends
    for ax_ind, offset, lbl, clr in zip([0, 0, 1, 1],
                                        [45, 20, 45, 20],
                                        [label, label2, label3, label4],
                                        [red, blue, orange, green]
                                        ):
        if lbl is not None:
            ax_joints[ax_ind].text(x_lim[1] - 10, y_lim[0] + offset, lbl, size=10, color=clr,
                                   horizontalalignment='right')
    # ax_joints[0].text(240, 20, 'Trained without 9s', size=10, color=blue, horizontalalignment='right')
    # ax_joints[1].text(240, 45, "Trained with 9s", size=10, color=orange, horizontalalignment='right')
    # ax_joints[1].text(240, 20, 'Trained without 9s', size=10, color=green, horizontalalignment='right')

    # plot diagonal lines
    for ax_ind in [0, 1]:
        ax_joints[ax_ind].plot(np.arange(*x_lim), np.arange(*y_lim), 'k--')
    if top_title is not None:
        ax_joints[0].set_title(top_title)
    if bottom_title is not None:
        ax_joints[1].set_title(bottom_title)
    if x_label is not None:
        ax_joints[1].set_xlabel(x_label)
    if y_label is not None:
        fig.text(0, 0.45, y_label, rotation="vertical")
    fig.tight_layout()
    return fig
