import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange
import matplotlib.pyplot as plt


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','.jpg','.jpeg'])

def load_image(filepath):
    img = Image.open(filepath)
    return img

class datasets_train(data.Dataset):
    def __init__(self,image_dir,upscale_factor,input_transform = None,target_transform=None):
        super(datasets_train, self).__init__()

        self.image_filenames = [join(image_dir,x) for x in listdir(image_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        target = load_image(self.image_filenames[index])
        input = target.copy()
        if self.input_transform:
            input = self.input_transform(input)

        if self.target_transform:
            target = self.target_transform(target)

        return input,target

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    p = datasets_train('./BSDS300/images/train',4)
    print(p.__getitem__(25)[0]) # mode=L gary image
    plt.figure(1)
    plt.imshow(p.__getitem__(25)[0])
    print(p.__getitem__(1)[0].size)
    plt.figure(2)
    plt.imshow(p.__getitem__(25)[1])
    print(p.__getitem__(1)[1].size)
    plt.show()

"""
model INPUT (80, 120)
model output (320,480)

datset label (321, 481)
"""










