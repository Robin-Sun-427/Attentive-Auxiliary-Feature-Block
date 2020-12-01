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
    #img = Image.open(filepath)
    img = Image.open(filepath).convert('YCbCr')
    y,cb,cr = img.split()
    return y

def get_patch(img_in,img_tar,patch_size,upscale_factor,ix=-1,iy=-1):
    #(ih,iw) -> LR size
    #(th,tw) -> expend LR for SR size
    (ih,iw) = img_in.size
    (th,tw) = (upscale_factor*ih, upscale_factor*iw)
    patch_mult = upscale_factor

    tp = patch_mult *patch_size#patch_size of HR
    ip = tp // upscale_factor# pactch_size LR

    if (ix == -1):
        ix = random.randrange(0,iw - ip + 1)
        iy = random.randrange(0,ih - ip + 1)
    (tx,ty) = (upscale_factor *ix, upscale_factor *iy)

    img_in = img_in.crop((iy,ix,iy+ip,ix+ip))
    img_tar = img_tar.crop((ty,tx,ty+tp,tx+tp))
    info_patch = {'ix':ix,'iy':iy,'ip':ip,'tx':tx,'ty':ty,'tp':tp}

    return img_in,img_tar,info_patch


class datasets_train(data.Dataset):
    def __init__(self,image_dir,patch_size,upscale_factor,input_transform = None,target_transform=None):
        super(datasets_train, self).__init__()

        self.image_filenames = [join(image_dir,x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        target = load_image(self.image_filenames[index])


        input = target.resize((int(target.size[0]//self.upscale_factor), int(target.size[1]//self.upscale_factor)),Image.BICUBIC)
        input,target, _ = get_patch(input,target,self.patch_size,self.upscale_factor)#(return img_in,img_tar,info_dict)


        if self.input_transform:
            input = self.input_transform(input)

        if self.target_transform:
            target = self.target_transform(target)

        return input,target

    def __len__(self):
        return len(self.image_filenames)











