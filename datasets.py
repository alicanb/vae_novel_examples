import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST


def _visualize(data, targets, num_images=100, num_cols=10, shuffle=True, **kwargs):
    num_rows = int(np.ceil(num_images / num_cols))
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))
    if shuffle:
        idxs = np.random.choice(targets.shape[0], num_images, replace=False)
    else:
        idxs = np.arange(num_images)
    axs = axs.flatten()
    for i, idx in enumerate(idxs):
        im = axs[i].imshow(data[idx], cmap='gray', **kwargs)
        axs[i].set_title('digit:{}'.format(targets[idx].item()))
        plt.colorbar(im, ax=axs[i])
        axs[i].axis('off')
    plt.show()


def _Holdout(DatasetClass, class_name=None):
    assert DatasetClass in [MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST], "Holdout doesn't work"

    class NewClass(DatasetClass):
        def __init__(self, root, targets=None, train=True, transform=None,
                     target_transform=None, download=False):
            super().__init__(root=root, train=train, transform=transform,
                             target_transform=target_transform, download=download)
            if targets is None:
                targets = list(self.class_to_idx.values())
            self.data = self.data[[elem in targets for elem in self.targets]]
            self.targets = self.targets[[elem in targets for elem in self.targets]]

        def visualize(self, num_images=100, num_cols=10, shuffle=True, **kwargs):
            _visualize(self.data, self.targets, num_images=num_images, num_cols=num_cols,
                       shuffle=shuffle, **kwargs)

        @property
        def raw_folder(self):
            return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'raw')

        @property
        def processed_folder(self):
            return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'processed')

    if class_name is None:
        class_name = "Holdout" + DatasetClass.__name__
    NewClass.__name__ = class_name
    return NewClass


def _Mini(DatasetClass, class_name=None):
    assert DatasetClass in [MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST], "Mini doesn't work"

    class NewClass(DatasetClass):
        def __init__(self, root, samples_per_target=10, train=True, transform=None,
                     target_transform=None, download=False, random=True, seed=None):
            super().__init__(root=root, train=train, transform=transform,
                             target_transform=target_transform, download=download)
            # TODO: Handle case for size > available data
            if random and seed is not None:
                np.random.seed(seed)
            ind = []
            for i in list(self.class_to_idx.values()):  # unique targets
                if random:
                    i_ind = np.random.choice(np.nonzero(self.targets == i).squeeze(), samples_per_target, replace=False)
                else:
                    i_ind = np.nonzero(self.targets == i).squeeze()[:samples_per_target]
                ind.append(torch.tensor(i_ind))
            ind = torch.cat(ind)
            self.data = self.data[ind]
            self.targets = self.targets[ind]

        def visualize(self, num_images=100, num_cols=10, shuffle=True, **kwargs):
            _visualize(self.data, self.targets, num_images=num_images, num_cols=num_cols,
                       shuffle=shuffle, **kwargs)

        @property
        def raw_folder(self):
            return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'raw')

        @property
        def processed_folder(self):
            return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'processed')

    if class_name is None:
        class_name = "Mini" + DatasetClass.__name__
    NewClass.__name__ = class_name
    return NewClass


HoldoutMNIST = _Holdout(MNIST)
MiniMNIST = _Mini(MNIST)

if __name__ == '__main__':
    a = MiniMNIST('.', samples_per_target=10)
    a.visualize(num_images=25, num_cols=5)
