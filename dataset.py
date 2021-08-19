import glob
import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T


class Cifar10(Dataset):
    def __init__(self, dataroot, transform=T.Compose([]), train=True, max_n_per_class=10000):
        self.dataroot_param = dataroot
        self.dataroot = dataroot
        self.train = train

        self.max_n_per_class = max_n_per_class

        self.transform = transform

        self.data = datasets.CIFAR10(root=self.dataroot, train=train, transform=self.transform, download=True)

        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.class_to_idx = self.data.class_to_idx
        self.idx_to_class = {self.class_to_idx[cls]:cls for cls in self.class_to_idx}

        self.subset_mask = np.array(self.data.targets)
        for i in range(10):
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
    
    def __getitem__(self, idx):
        idx_in_ori_data = self.subset_indices[idx]
        return self.data.__getitem__(idx_in_ori_data)

    def __len__(self):
        return len(self.subset_indices)

    def __repr__(self):
        repr = """Cifar10 Dataset(subset):
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class ImageNet(Dataset):
    """ ImageNet dataset with subset and MAX #data-per-class settings.
    If use default parameters, it will just return a dataset with all ImageNet data.
    Otherwise, it will return a subset of ImageNet dataset.
    """
    def __init__(self, dataroot, transform=T.Compose([]), train=True, subset=1000, max_n_per_class=10000):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "ilsvrc2012")
        self.train = train

        # Class number of subset. Take top-N classes as a subset(Torchvision official implementation sorting).
        self.subset = subset
        # Max number of data per class. If it was set more than the total number of that class, all the data will be taken.
        # Otherwise, it will take top-N data of that class(Torchvision official implementation sorting).
        self.max_n_per_class = max_n_per_class

        self.transform = transform
        
        self.data = datasets.ImageNet(root=self.dataroot, split='train' if train else 'val', transform=self.transform)
        
        # Metadata of dataset
        self.classes = self.data.classes
        self.class_num = len(self.data.classes)
        self.idx_to_class = {i:self.data.classes[i][0] for i in range(self.class_num)}
        self.class_to_idx = {}
        for i in self.idx_to_class:
            classname = self.idx_to_class[i]
            while classname in self.class_to_idx:
                classname += '_'
            self.class_to_idx[classname] = i
        
        # Subset process.
        if isinstance(subset, int):
            self.class_subset = list(range(subset))
        else:
            self.class_subset = list(subset)

        self.mapping = {i: self.class_subset[i] for i in range(len(self.class_subset))}
        self._rev_mapping = {self.mapping[i]: i for i in self.mapping}
        self._rev_mapping = np.array([self._rev_mapping[i] if i in self.class_subset else -1 for i in range(1000)])
        target_mapping = lambda x: self._rev_mapping[x]

        self.subset_mask = np.array(self.data.targets)
        for i in self.class_subset:
            self.subset_mask[np.where(self.subset_mask == i)[0][self.max_n_per_class:]] = -1
        self.subset_indices = np.where(self.subset_mask != -1)[0]
        self.class_selection = np.where(np.in1d(np.array(self.data.targets), np.array(self.class_subset)) == 1)[0]
        self.subset_indices = np.intersect1d(self.subset_indices, self.class_selection)

        # Data and targets
        self.targets = list(target_mapping(np.array(self.data.targets)[self.subset_indices]))
        self.img_paths = list(np.array(self.data.imgs)[self.subset_indices][:, 0])

        # Metadata override.
        self.classes = [self.classes[i] for i in self.class_subset]
        self.class_num = len(self.classes)
        self.idx_to_class = {i: self.idx_to_class[i] for i in self.class_subset}
        self.class_to_idx = {self.idx_to_class[i]: i for i in self.idx_to_class}


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr
