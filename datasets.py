import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets


def _Holdout(DatasetClass, class_name=None):
    assert DatasetClass in [datasets.MNIST, datasets.EMNIST, datasets.FashionMNIST], "Holdout doesn't work"

    class NewClass(DatasetClass):
        def __init__(self, root, targets=None, train=True, transform=None,
                     target_transform=None, download=False):
            # TODO: fix this so you don't download dataset again
            super().__init__(root=root, train=train, transform=transform,
                             target_transform=target_transform, download=download)
            if targets is None:
                targets = list(self.class_to_idx.values())
            self.data = self.data[[elem in targets for elem in self.targets]]
            self.targets = self.targets[[elem in targets for elem in self.targets]]

        def visualize(self, num_images=100, num_cols=10, shuffle=True, **kwargs):
            num_rows = int(np.ceil(num_images / num_cols))
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))
            if shuffle:
                idxs = np.random.choice(self.targets.shape[0], num_images, replace=False)
            else:
                idxs = np.arange(num_images)
            axs = axs.flatten()
            for i, idx in enumerate(idxs):
                im = axs[i].imshow(self.data[idx], cmap='gray', **kwargs)
                axs[i].set_title('digit:{}'.format(self.targets[idx].item()))
                plt.colorbar(im, ax=axs[i])
                axs[i].axis('off')

            plt.show()

    if class_name is None:
        class_name = "Holdout" + DatasetClass.__name__
    NewClass.__name__ = class_name
    return NewClass


HoldoutMNIST = _Holdout(datasets.MNIST)

if __name__ == '__main__':
    a = HoldoutMNIST('.', targets=[1, 2])
    a.visualize(num_images=25, num_cols=5)
