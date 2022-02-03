import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os.path

class MNIST_cutout(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, cutout = None, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.data, self.lebels = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        # print(self.data.shape)
        # print(self.lebels.shape)

        if cutout != None:
            self.data = torch.cat((self.data[self.lebels != cutout], self.data[self.lebels == cutout][:600,:]))
            self.lebels = torch.cat((self.lebels[self.lebels != cutout], self.lebels[self.lebels == cutout][:600]))
            # print(self.data.shape)
            # print(self.lebels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.lebels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))