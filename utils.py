import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.optim.adam import Adam

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split # using for split training set and val set
from torch.utils.tensorboard import SummaryWriter

#import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_image(img_tensor, label):
    print('Label:', dataset.classes[label], '(' + str(label) + ')')
    img_tensor = denormalize(img_tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))[0].permute((1, 2, 0))
    plt.imshow(img_tensor)

def get_train_test_dataset(dataset, train_ratio=0.8):
    random_seed = 43
    torch.manual_seed(random_seed)
    train_size = int(len(dataset)*train_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds, test_ds = random_split(val_ds, [int(len(val_ds)*0.5), len(val_ds)-int(len(val_ds)/2)])
    
    return train_ds, val_ds, test_ds

#utils function
def acc_metric(pred, label):
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    label = label.cpu().numpy()
    
    acc = np.array((label == pred)).sum()/len(label)
    return acc

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

def test_predict_image(img, model, device="cuda"):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]
