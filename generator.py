import os
import logging
from model.unet_model import UNet
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

import glob
import shutil
import cv2

#K1
img_path_pattern = "xxx" # used to guide original train_image path
label_path_pattern = "xxx" # used to guide original train_label path

img_paths = glob.glob(img_path_pattern)
label_paths = glob.glob(label_path_pattern)

img_paths.sort()
label_paths.sort()

img_label_pairs = list(zip(img_paths, label_paths))

train_img_dir = "xxx" # used to save train_image
train_label_dir = "xxx" # used to save train_label
val_img_dir = "xxx" # used to save val_image
val_label_dir = "xxx" # used to save val_label
test_img_dir = "xxx" # used to save test_image
test_label_dir = "xxx" # used to save test_label

for directory in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
    os.makedirs(directory, exist_ok=True)

# train dataset self-segmentation
untrain_percent = 0.2
n_untrain = int(len(img_label_pairs) * untrain_percent)
n_val = int(n_untrain / 2)

n_test = n_untrain - n_val
n_train = len(img_label_pairs) - n_untrain

train_set, untrain_set = random_split(img_label_pairs, [n_train, n_untrain], generator=torch.Generator().manual_seed(0))
val_set, test_set = random_split(untrain_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))


def save_set(dataset, img_dir, label_dir):
    count = 0
    for img_path, label_path in dataset:
        count += 1
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        
        cv2.imwrite(os.path.join(img_dir, f'{count}.png'), img)
        cv2.imwrite(os.path.join(label_dir, f'{count}.png'), label)

save_set(train_set, train_img_dir, train_label_dir)
save_set(val_set, val_img_dir, val_label_dir)
save_set(test_set, test_img_dir, test_label_dir)