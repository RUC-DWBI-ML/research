import os
import math
import random
import torch
import numpy
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def gcn(images, multiplier=55, eps=1e-10):
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    images = multiplier * images / per_image_norm
    return images

def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)





class Cifar10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=None)
        self.data = {}
        self.data['images'] = self.cifar10.data
        self.data['labels'] = np.array(self.cifar10.targets)
        self.data['images'] = gcn(self.data['images'])
        mean, zca_decomp = get_zca_normalization_param(self.data['images'])
        self.data['images'] = zca_normalization(self.data['images'], mean, zca_decomp)
        self.data["images"] = np.transpose(self.data["images"], (0, 3, 1, 2))

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.data['images'][index],self.data['labels'][index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class Cifar10_test(Dataset):
    def __init__(self, path):
        self.cifar10_test = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=False,
                                        transform=None)
        self.data = {}
        self.data['images'] = self.cifar10_test.data
        self.data['labels'] = np.array(self.cifar10_test.targets)
        self.data['images'] = gcn(self.data['images'])
        mean, zca_decomp = get_zca_normalization_param(self.data['images'])
        self.data['images'] = zca_normalization(self.data['images'], mean, zca_decomp)
        self.data["images"] = np.transpose(self.data["images"], (0, 3, 1, 2))

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.data['images'][index], self.data['labels'][index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10_test)


class Cifar100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=None)
        self.data = {}
        self.data['images'] = self.cifar100.data
        self.data['labels'] = np.array(self.cifar100.targets)
        self.data["images"] = np.transpose(self.data["images"], (0, 3, 1, 2))

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.data['images'][index], self.data['labels'][index]
        return data,target,index

    def __len__(self):
        return len(self.cifar100)


class Cifar100_test(Dataset):
    def __init__(self, path):
        self.cifar100_test = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=False,
                                        transform=None)
        self.data = {}
        self.data['images'] = self.cifar100_test.data
        self.data['labels'] = np.array(self.cifar100_test.targets)
        self.data["images"] = np.transpose(self.data["images"], (0, 3, 1, 2))

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.data['images'][index], self.data['labels'][index]
        return data, target, index

    def __len__(self):
        return len(self.cifar100_test)



def resize_transorm(x):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
    ])
    return transform(x)

