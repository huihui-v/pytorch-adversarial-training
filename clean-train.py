import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from tqdm.auto import tqdm

# from methods import PGD, Baseline, Caltech256
from modules import PGD, Baseline
from dataset import Caltech256



def criterion(output, target):
    loss = nn.functional.cross_entropy(output, target)
    return loss

def evaluation(model, loader, device):
    model.eval()
    running_loss = 0.
    running_tp = 0
    running_total = 0
    processbar = tqdm(total=len(loader), leave=False)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = criterion(output, target)
        running_loss += loss.item()

        pred = output.argmax(dim=1)
        tp = (pred == target).sum().item()
        total = pred.shape[0]
        running_tp += tp
        running_total += total

        processbar.update(1)

    processbar.close()
    mean_loss  = running_loss / len(loader)
    mean_acc = 1. * running_tp / running_total

    return (mean_loss, mean_acc)

def evaluation_adv(model, attacker, loader, device):
    model.eval()
    running_loss = 0.
    running_tp = 0
    running_total = 0
    processbar = tqdm(total=len(loader), leave=False)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        atk_data = attacker.attack(data, target)
        output = model(atk_data)

        loss = criterion(output, target)
        running_loss += loss.item()

        pred = output.argmax(dim=1)
        tp = (pred == target).sum().item()
        total = pred.shape[0]
        running_tp += tp
        running_total += total

        processbar.update(1)

    processbar.close()
    mean_loss  = running_loss / len(loader)
    mean_acc = 1. * running_tp / running_total

    return (mean_loss, mean_acc)

def run(lr, max_epoch, batch_size, dataroot, device_id):
    device = torch.device(device_id)

    train_transforms = T.Compose([
        T.Resize((256, 256), interpolation=Image.NEAREST),
        T.RandomResizedCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transforms = T.Compose([
        T.Resize((224, 224), interpolation=Image.NEAREST),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_dataset = Caltech256(dataroot, train=True, transforms=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    test_dataset = Caltech256(dataroot, train=False, transforms=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    train_test_dataset = Caltech256(dataroot, train=True, transforms=test_transforms)
    train_test_dataloader = DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

    model = Baseline(train_dataset.class_num)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
    atk = PGD(model, epsilon=8/255, step=2/255, iterations=10, random_start=True)

    prev = 100.
    for epoch_idx in range(max_epoch):
        model.train()

        running_loss = 0.
        processbar = tqdm(total=len(train_dataloader), leave=False)
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = criterion(output, target)
            running_loss += loss.detach().item()
            processbar.set_postfix({"Loss r.": loss.detach().item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processbar.update(1)
        processbar.close()
        mean_loss = running_loss / len(train_dataloader)

        eval_trainset_loss, eval_trainset_acc = evaluation(model, train_test_dataloader, device)
        eval_testset_loss, eval_testset_acc = evaluation(model, test_dataloader, device)
        eval_rtrainset_loss, eval_rtrainset_acc = evaluation_adv(model, atk, train_test_dataloader, device)
        eval_rtestset_loss, eval_rtestset_acc = evaluation_adv(model, atk, test_dataloader, device)

        scheduler.step()

        print(f"Finish {epoch_idx}!")
        print("Train.acc {:.4f}, test.acc {:.4f}, rtrian.acc {:.4f}, rtest.acc {:.4f}".format(eval_trainset_acc, eval_testset_acc, eval_rtrainset_acc, eval_rtestset_acc))
        print("Train.l {:.4f}, test.l {:.4f}, rtrian.l {:.4f}, rtest.l {:.4f}".format(eval_trainset_loss, eval_testset_loss, eval_rtrainset_loss, eval_rtestset_loss))

        if (abs(eval_testset_loss - prev) < 1e-4):
            print("Early stop")
            break
        prev = eval_testset_loss

    torch.save(model.cpu(), 'test.pth')
    print('Save model.')


if __name__ == '__main__':
    lr = 1e-3
    max_epoch = 120
    batch_size = 64
    dataroot = os.path.join(os.environ["DATAROOT"], "Caltech-256")
    # dataroot = '/home/huihui/Projects/caltech-256/Caltech-256'
    device_id = "cuda:0"
    run(lr, max_epoch, batch_size, dataroot, device_id)