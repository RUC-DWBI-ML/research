import os
import numpy as np
import torch
import math
import random
import numpy
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from contrast.utils.utils import set_random_seed

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



CIFAR10_SUPERCLASS = list(range(10))  
IMAGENET_SUPERCLASS = list(range(30))  

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],#1
    [1, 33, 67, 73, 91],#2
    [54, 62, 70, 82, 92],#3
    [9, 10, 16, 29, 61],#4
    [0, 51, 53, 57, 83],#5
    [22, 25, 40, 86, 87],#6
    [5, 20, 26, 84, 94],#7
    [6, 7, 14, 18, 24],#8
    [3, 42, 43, 88, 97],#9
    [12, 17, 38, 68, 76],#10
    [23, 34, 49, 60, 71],#11
    [15, 19, 21, 32, 39],#12
    [35, 63, 64, 66, 75],#13
    [27, 45, 77, 79, 99],#14
    [2, 11, 36, 46, 98],#15
    [28, 30, 44, 78, 93],#16
    [37, 50, 65, 74, 80],#17
    [47, 52, 56, 59, 96],#18
    [8, 13, 48, 58, 90],#19
    [41, 69, 81, 85, 89],#20
]

class Cifar10_train(Dataset):
    def __init__(self, path,download,trans):
        self.cifar10 = datasets.CIFAR10(path, train=True, download=download, transform=trans)
    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.cifar10[index]
        return data,target,index
    def __len__(self):
        return len(self.cifar10)

class Cifar10_test(Dataset):
    def __init__(self, path,download,trans):
        self.cifar10 = datasets.CIFAR10(path, train=False, download=download, transform=trans)
    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.cifar10[index]
        return data,target,index
    def __len__(self):
        return len(self.cifar10)


class Cifar100_train(Dataset):
    def __init__(self, path,download,trans):
        self.cifar100 = datasets.CIFAR100(path, train=True, download=download, transform=trans)
    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.cifar100[index]
        return data,target,index
    def __len__(self):
        return len(self.cifar100)



class Cifar100_test(Dataset):
    def __init__(self, path,download,trans):
        self.cifar100 = datasets.CIFAR100(path, train=False, download=download, transform=trans)
    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.cifar100[index]
        return data,target,index
    def __len__(self):
        return len(self.cifar100)


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_contrastive_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False):
    train_transform, test_transform = get_transform(image_size=image_size)
    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = Cifar10_train(P.data_path,download,train_transform)
        test_set = Cifar10_test(P.data_path, download, test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = Cifar100_train(P.data_path, download, train_transform)
        test_set = Cifar100_test(P.data_path, download, test_transform)
    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def Get_initial_dataset(dataset, classes, budget):

    target_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] in classes]
    set_random_seed(0)
    initial_indices = random.sample(target_index, budget)
    initial_dataset = Subset(dataset, initial_indices)
    return initial_dataset, initial_indices


def get_sub_test_dataset(dataset, classes):
    labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] in classes]
    random.shuffle(labeled_index)
    dataset_test = Subset(dataset, labeled_index)
    return dataset_test, labeled_index


def get_sub_unlabeled_dataset(dataset, select_L_index,select_O_index, target_list, num_images):
    all_index = set(np.arange(num_images))
    select_index = select_L_index + select_O_index
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)
    datasey_UL = Subset(dataset, unlabeled_L_index)
    datasey_UO = Subset(dataset, unlabeled_O_index)
    dataset_U = Subset(dataset, unlabeled_indices)

    return dataset_U, datasey_UL, datasey_UO, unlabeled_indices, unlabeled_L_index, unlabeled_O_index



def Get_mismatch_unlabeled_dataset(dataset, initial_index, target_list, mismatch, num_images):
    """
    Build the unlabeled dataset according to the mismatch proportion.
    The unlabeled dataset contains the following:
    (1) all the instances from target categories besides the labeled ones;
    (2) the randomly selected instances with unknown categories based on the mismatch proportion.
    """
    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),initial_index)) 
    unlabeled_T_index = []
    unlabeled_U_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_T_index.append(i)
        else:
            unlabeled_U_index.append(i)

    target_number = len(unlabeled_T_index)
    unknown_number = math.ceil((mismatch*target_number)/(1-mismatch))

    set_random_seed(0)
    select_U_index = random.sample(unlabeled_U_index, unknown_number)
    unlabeled_index = unlabeled_T_index + select_U_index
    dataset_U = Subset(dataset, unlabeled_index)
    return dataset_U,unlabeled_index


def get_mismatch_class_unlabeled_dataset(dataset, select_L_index, num_images):

    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_L_index))  # find indices which is in all_indices but not in current_indices

    set_random_seed(0)
    dataset_U = Subset(dataset, unlabeled_indices)

    return dataset_U,unlabeled_indices



def Get_mismatch_contrast_dataset(dataset, select_L_index, target_list,mismatch, num_images):
    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_L_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)

    target_number = len(unlabeled_L_index)
    others_number = math.ceil((mismatch*target_number)/(1-mismatch))

    set_random_seed(0)
    select_O_index = random.sample(unlabeled_O_index, others_number)
    unlabeled_index = unlabeled_L_index + select_O_index
    contrast_index = unlabeled_index + select_L_index

    set_random_seed(0)
    random.shuffle(contrast_index)
    dataset_contrast = Subset(dataset, contrast_index)

    return dataset_contrast,contrast_index



def get_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False):
    train_transform, test_transform = get_transform(image_size=image_size)
    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = Cifar10_train(P.data_path,download,train_transform)
        test_set = Cifar10_test(P.data_path, download, test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = Cifar100_train(P.data_path, download, train_transform)
        test_set = Cifar100_test(P.data_path, download, test_transform)
    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


def Get_dataloader(train_set, sample_index, args, group_index=[], group=False, ssl=False):
    """
    Build the dataloader for training
    """  
    if ssl:
        sampler = data.sampler.SubsetRandomSampler(sample_index)
        dataloader = data.DataLoader(train_set, sampler=sampler, batch_size=args.batch_size_classifier, drop_last = False)
        return dataloader
    else:
        if group:
            group_loader = []
            for i in range(len(group_index)):
                sampler_group = data.sampler.SubsetRandomSampler(group_index[i])  # make indices initial to the samples
                loader = data.DataLoader(train_set, sampler=sampler_group,batch_size=args.con_batch_size)
                group_loader.append(loader)
            return group_loader
        else:
            sampler = data.sampler.SubsetRandomSampler(sample_index)
            dataloader = data.DataLoader(train_set, sampler=sampler,batch_size=args.con_batch_size)
            return dataloader





def Get_group_index(dataset, L_index,args):
    label_i_index = [[] for i in range(len(args.target_list))]
    for i in L_index:
        for k in range(len(args.target_list)):
            if dataset[i][1] == args.target_list[k]:
                label_i_index[k].append(i)
    return label_i_index

def Get_group_index_train(labels, L_index, args):
    label_i_index = [[] for i in range(len(args.target_list))]
    for i in range(len(L_index)):
        for k in range(len(args.target_list)):
            if labels[i] == args.target_list[k]:
                label_i_index[k].append(L_index[i])
    return label_i_index


def shuffles(index):
    seed_torch()
    random.shuffle(index)
    return index






# correct the label. For example, the original label is [2,6,9], the correct label is [0,1,2]
def rectify_labels(labels,args):
    rectify_label = labels
    for i in range(len(labels)):
        for k in range(len(args.target_list)):
            if labels[i] == args.target_list[k]:
                rectify_label[i] = k
    return rectify_label


def polynomial_decay(current_step, max_learning_rate, min_learning_rate, decay_steps, power=2):
    if current_step > decay_steps:
        current_step = decay_steps
    decayed_learning_rate = (max_learning_rate - min_learning_rate) * (1 - float(current_step) / float(decay_steps))**(power) + min_learning_rate
    return decayed_learning_rate