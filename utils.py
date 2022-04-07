import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import random_split

from torch.optim.adam import Adam
import cv2




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
    torch.manual_seel(random_seed)
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