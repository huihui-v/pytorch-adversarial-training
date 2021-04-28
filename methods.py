import glob
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset



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

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.bone = models.resnet18(pretrained=True)
        self.bone.fc = nn.Linear(in_features=512, out_features=257, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        output = self.model(x)
        return output

class PGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, random_start=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(PGD, self).__init__()
        # Arguments of PGD
        self.model = model
        self.device = next(model.parameters()).device
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.mean = mean
        self.std = std
        # Normalized clip minimum and maximum
        self.clip_min = [(0-mean[i])/std[i] for i in range(3)]
        self.clip_max = [(1-mean[i])/std[i] for i in range(3)]
        # Model status
        self.training = self.model.training

    def clip_perturbation(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        perturbation = torch.clip(perturbation, -self.epsilon, self.epsilon)
        return perturbation

    def compute_perturbation(self, adv_x, x):
        # Clamp the adversarial image to a legal 'image'
        for i in range(3):
            adv_x[:, i] = torch.clamp(adv_x[:, i], min=self.clip_min[i], max=self.clip_max[i])
        perturbation = adv_x - x
        # Clamp the perturbation to epsilon
        perturbation = self.clip_perturbation(perturbation)

        return perturbation

    def onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        criterion = nn.functional.cross_entropy
        output = self.model(adv_x)
        atk_loss = criterion(output, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        next_perturbation = self.step*torch.sign(grad)
        adv_x = adv_x.detach() + next_perturbation
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    # def attack(self, x, target):
    #     x = x.to(self.device)
    #     target = target.to(self.device)
    #     loss = nn.CrossEntropyLoss()
            
    #     ori_x = x.data
            
    #     self.model.eval()
    #     self._model_freeze()
    #     for i in range(self.iterations) : 
    #         x.requires_grad = True
    #         outputs = self.model(x)

    #         self.model.zero_grad()
    #         cost = loss(outputs, target).to(self.device)
    #         cost.backward()
    #         # Essential
    #         x.requires_grad = False

    #         adv_x = x + self.step*torch.sign(x.grad)
    #         eta = torch.clamp(adv_x - ori_x, min=-self.epsilon, max=self.epsilon)
    #         for i in range(3):
    #             x[:, i] = torch.clamp((ori_x + eta)[:, i], min=self.clip_min[i], max=self.clip_max[i])
    #         x = x.detach()
                
    #     self._model_unfreeze()
    #     if self.training:
    #         self.model.train()

    #     return x

    def attack1(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        self.model.eval()
        self._model_freeze()
        perturbation = torch.zeros_like(x).to(self.device)
        for i in range(self.iterations):
            perturbation = self.onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation