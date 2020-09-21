import os,sys

import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
import torchvision.transforms as T 

import PIL.Image as Image

from utils import test_and_add_postfix_dir

def get_mnist(path, batch_size, num_workers, input_size=64):

    if not isinstance(input_size, tuple):
        input_size = (input_size, input_size)

    data = MNIST(path, train=True, transform=T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        # T.Normalize([0,5], [0.5])
        T.Normalize((0.5,),(0.5,))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

def get_cifar10(path, batch_size, num_workers, input_size=64):

    if not isinstance(input_size, tuple):
        input_size = (input_size, input_size)

    data = CIFAR10(path, train=True, transform=T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

def get_fashion(path, batch_size, num_workers, input_size=64):

    if not isinstance(input_size, tuple):
        input_size = (input_size, input_size)

    data = FashionMNIST(path, train=True, transform=T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

def get_svhn(path, batch_size, num_workers, input_size=64):

    if not isinstance(input_size, tuple):
        input_size = (input_size, input_size)

    data = SVHN(path, split="train", transform=T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class UnlabeledImageFolder(Dataset):
    """ Unlabeled image data under one folder
    """
    def __init__(self, root, transform=None, compatible=True):
        self.root = test_and_add_postfix_dir(root)
        self.images_path = self.root 
        self.filenames = os.listdir(self.images_path)
        self.len = len(self.filenames)
        self.transform = transform
        self.compatible = compatible # compatible with training method using labels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        name = self.filenames[idx]
        image = Image.open(self.images_path + name)

        if self.transform:
            image = self.transform(image)
        if self.compatible:
            return image, 0
        return image 


def get_unlabeled_celebA(path, batch_size, num_workers, input_size=64):

    if not isinstance(input_size, tuple):
        input_size = (input_size, input_size)

    data = UnlabeledImageFolder(path, transform=T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


# 对 1024x1024 的图像读取再 resize 不知道为什么要这么多时间？ > 1s
class Resize:
    def __init__(self, size):

        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size

    def __call__(self, image: Image.Image):
        return image.resize(self.size)


def get_folder_dataset(path, batch_size, num_workers, input_size=256):

    if not isinstance(input_size, tuple):
        input_size = (input_size, input_size)

    data = UnlabeledImageFolder(path, transform= 
    # None)
    T.Compose([
        T.Resize(input_size, Image.NEAREST),
        # Resize(input_size),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

if __name__ == '__main__':

    path =  "/home/victorchen/workspace/Venus/torch_download/svhn"
    get_svhn(path, 32, 4)
