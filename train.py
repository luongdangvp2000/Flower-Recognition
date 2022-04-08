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

from model import (
    VGG16,
    ResNet9,
)

from dataset import (
    FlowersDataset,
)

from utils import(
    save_checkpoint,
    load_checkpoint,
    denormalize,
    show_image,
    get_train_test_dataset,
    acc_metric,
)

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
IMAGE_HEIGHT = 220
IMAGE_WIDTH = 220
NUM_WORKERS = 1
ROOT_DIR = 'data/flowers'
LOAD_MODEL = False
writer = SummaryWriter(f'runs/Flower-Recognition/tryingout_tensorboard')
# step = 0

def train(model, epoch, train_loader, valid_loader, loss_fn, acc_metric, optimizer, device, step=0):
    for i in range(epoch):
        print(f'epoch {i+1}/{epoch}')
        model = model.to(device)
        # train
        model.train()
        losses = []
        acc = 0
        for train_input, train_label in (train_loader):
            train_input, train_label = train_input.to(device), train_label.to(device).squeeze()
            train_output = model(train_input)
            loss = loss_fn(train_output, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            acc += acc_metric(train_output, train_label)

            img_grid = torchvision.utils.make_grid(train_input)
            writer.add_image('Flowers Recognition Images', img_grid)

            #Add traning loss and accuracy to tensorboard
            writer.add_scalar("Training Loss", loss, global_step=step)
            writer.add_scalar("Training Accuracy", acc, global_step=step)
            step += 1

        print(acc, len(train_loader))
        acc = (acc / len(train_loader)) * 100
        print(f'train loss: {np.mean(losses)} and train acc: {acc}')

        # Doing hyperparamenters search on tensorboard
        writer.add_hparams(
            {'Learning rate': LEARNING_RATE, 'Batch size': BATCH_SIZE},
            {'Accuracy': acc, 'Loss': np.mean(losses)}
        )

        model.eval()
        eval_losses = []
        eval_acc = 0
        for eval_input, eval_label in (valid_loader):
            eval_input, eval_label = eval_input.to(device), eval_label.to(device).squeeze()
            eval_output = model(eval_input)
            loss = loss_fn(eval_output, eval_label)

            eval_losses.append(loss.item())
            eval_acc += acc_metric(eval_output, eval_label)

        eval_acc = eval_acc / len(valid_loader) * 100
        print(f'valid loss: {np.mean(eval_losses)} and valid acc: {eval_acc}')
        
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        #print("=> Saving checkpoint")
        save_checkpoint(checkpoint)
        print('*******')

def main():
    transforms = A.Compose([
        A.Resize(IMAGE_HEIGHT,IMAGE_HEIGHT),
        #A.RandomCrop(64, padding=4, padding_mode='reflect'),
        A.HorizontalFlip(),
        A.Rotate(10),
        A.ColorJitter(brightness=0.1, contrast=0.1,saturation=0.1,hue=0.1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    #set up data
    dataset = FlowersDataset(root_dir = ROOT_DIR, transforms = transforms)
    train_ds, val_ds, test_ds = get_train_test_dataset(dataset, train_ratio=0.8)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    #model
    model = ResNet9(in_channels=3, n_classes=5)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, NUM_EPOCHS, train_loader, valid_loader, loss_fn, acc_metric, optimizer, DEVICE)

    #writer.flush()
if __name__ == "__main__":
    main()