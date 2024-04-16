import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(0.330, 0.204)
    transforms.Normalize(0.5, 0.5)
])

class Loader(Dataset):
    def __init__(self, data_path):
        # print(data_path)
        images = []
        image_root = os.path.join(data_path,'train_img')

        # print(image_root)
        for image_name in os.listdir(image_root):
          image_path = os.path.join(image_root, image_name)
          images.append(image_path)

        self.data_path = data_path
        self.imgs_path = images

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('train_img', 'train_label')

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

        if label.max() > 1:
            label = label / 255

        image = transform(image)

        return image, label

    def __len__(self):
        return len(self.imgs_path)



if __name__ == "__main__":

    path='xxx' # dir used to store train, val and test dataset
    dataset = Loader(path)
    print("data_num:", len(dataset))

    image, label = dataset[0]
    print(image.shape)
    print(label.shape)
    print(image.max())
    print(label.max())