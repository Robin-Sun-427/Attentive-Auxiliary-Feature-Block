from os.path import join
from torchvision.transforms import Compose,CenterCrop,ToTensor,Resize
from datasets import datasets_train


def input_transform():
    return Compose([ToTensor(),])
def target_transform():
    return Compose([ToTensor(),])


#对读取的图像进行修剪值相同尺寸
def calculate_crop_size(crop_size,upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_trainning_set(patch_size,upscale_factor):
    root_dir = './BSDS300/images'
    train_dir = join(root_dir,'train')
    return datasets_train(train_dir,patch_size,upscale_factor,input_transform=input_transform(),
                          target_transform=target_transform())

def get_eval_set(patch_size,upscale_factor):
    root_dir = './BSDS300/images'
    test_dir = join(root_dir,'test')
    return datasets_train(test_dir, patch_size,upscale_factor, input_transform=input_transform(),
                          target_transform=target_transform())
