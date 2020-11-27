
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
from AAF_Network import AAF

# Prediction Demo parameters Setting
parser = argparse.ArgumentParser(description='AAF model SR results')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--scale_factor', type=float, help='factor by which super resolution needed')
opt = parser.parse_args()

img = Image.open(opt.input_image)

model = AAF(3,64,4,4) # AAF(input_channels,base_filters,num_stages,upscale_factor)
model.load_state_dict(torch.load(opt.model))

input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
out = model(input)# torch.Tensor

"""
torch.Tensor convert to PIL Image 
tt = transforms.ToPILImage()
img_out = tt(out.data[0])

"""
tt = transforms.ToPILImage()
img_out = tt(out.data[0])# out shape channel is 3
img_out.save(opt.output_filename)


plt.figure()
plt.imshow(img_out)
plt.title('SR process result)
plt.figure()
plt.imshow(img)
plt.title('input LR image')
plt.show()
