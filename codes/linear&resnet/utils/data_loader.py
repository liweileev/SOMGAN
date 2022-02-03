import torch
import torchvision.datasets as dsets
from torchvision import transforms
from utils.MNIST_cutout import MNIST_cutout

class Data_Loader():
    def __init__(self, dataset, image_path, image_size, batch_size, shuf=True, cutout=None):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.cutout = cutout

    def transform(self, centercrop=False, resize=True, totensor=True, normalize=True):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='bedroom_train'):
        transforms = self.transform(centercrop=False, resize=True, totensor=True, normalize=True)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset
    
    def load_cifar(self):
        transforms = self.transform(centercrop=False, resize=True, totensor=True, normalize=True)
        dataset = dsets.CIFAR10(self.path, transform=transforms)
        return dataset
    
    def load_mnist(self):
        transforms = self.transform(centercrop=False, resize=True, totensor=True, normalize=True)
        dataset = dsets.MNIST(self.path, transform=transforms)
        return dataset
        
    def load_mnist_cutout(self):
        transforms = self.transform(centercrop=False, resize=True, totensor=True, normalize=True)
        dataset = MNIST_cutout(self.path, cutout = self.cutout, transform=transforms)
        return dataset

    def load_stl10(self):
        transforms = self.transform(centercrop=False, resize=True, totensor=True, normalize=True)
        dataset = dsets.STL10(self.path, split='train+unlabeled' ,transform=transforms)
        return dataset

    def load_folder(self):
        transforms = self.transform(centercrop=True, resize=True, totensor=True, normalize=True)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset
    
    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'cifar':
            dataset = self.load_cifar()
        elif self.dataset == 'mnist':
            dataset = self.load_mnist()
        elif self.dataset == 'stl10':
            dataset = self.load_stl10()
        elif self.dataset == 'mnist_cutout':
            dataset = self.load_mnist_cutout()
        else:
            dataset = self.load_folder()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader

