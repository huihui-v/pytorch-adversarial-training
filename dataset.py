import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class Caltech256(Dataset):
    """Dataset Caltech 256
    Class number: 257
    Train data number: 24582
    Test data number: 6027
    """
    def __init__(self, dataroot, transforms=None, train=True):
        # Initial parameters
        self.dataroot = dataroot
        self.train = train
        if transforms: # Set default transforms if no transformation provided.
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])
        
        # Metadata of dataset
        classes = [i.split('/')[-1] for i in glob.glob(os.path.join(dataroot, 'data', '*'))]
        self.class_num = len(classes)
        self.classes = [i.split('.')[1] for i in classes]
        self.class_to_idx = {i.split('.')[1]: int(i.split('.')[0])-1 for i in classes}
        self.idx_to_class = {int(i.split('.')[0])-1: i.split('.')[1] for i in classes}
        
        # Split file and image path list.
        self.split_file = os.path.join(dataroot, 'trainset.txt') if train else os.path.join(dataroot, 'testset.txt')
        with open(self.split_file, 'r') as f:
            self.img_paths = f.readlines()
            self.img_paths = [i.strip() for i in self.img_paths]
        self.targets = [self.class_to_idx[i.split('/')[1].split('.')[1]] for i in self.img_paths]
        self.img_paths = [os.path.join(dataroot, i) for i in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transforms(img)
        target = self.targets[idx]

        return (img_tensor, target)

    def __repr__(self):
        repr = """Caltech-256 Dataset:
\tClass num: {}
\tData num: {}""".format(self.class_num, self.__len__())
        return repr
