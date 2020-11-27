from os.path import join
from torchvision.transforms import Compose,CenterCrop,ToTensor,Resize
from datasets import datasets_train



def input_transform(croped_size,upscale_factor):
    return Compose([CenterCrop(croped_size),
                    Resize((croped_size//upscale_factor, croped_size//upscale_factor)),
                    ToTensor(),])

def target_transform(croped_size):
    return Compose([CenterCrop(croped_size),
                    ToTensor(),])


#对读取的图像进行修剪值相同尺寸
def calculate_crop_size(crop_size,upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_trainning_set(upscale_factor):
    root_dir = './BSDS300/images'
    train_dir = join(root_dir,'train')
    croped_size = calculate_crop_size(256,upscale_factor)

    return datasets_train(train_dir,upscale_factor,input_transform=input_transform(croped_size,upscale_factor),
                          target_transform=target_transform(croped_size))

def get_eval_set(upscale_factor):
    root_dir = './BSDS300/images'
    test_dir = join(root_dir,'test')
    croped_size = calculate_crop_size(256,upscale_factor)
    return datasets_train(test_dir, upscale_factor, input_transform=input_transform(croped_size, upscale_factor),
                          target_transform=target_transform(croped_size))
