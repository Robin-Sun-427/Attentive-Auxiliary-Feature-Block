import torch
import torch.nn as nn
from AAF_base_network import *
from torchsummary import summary
import torch.nn.functional as F


class AAF(nn.Module):
    def __init__(self,num_channels,base_filter,num_stages,scale_factor):
        super(AAF, self).__init__()

        self.scale_factor = scale_factor

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding =2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        self.head_layer = Conv_Block(num_channels,base_filter,3,1,1,'relu')
        #ksize=3,strides=1,padding=1 是保存卷积前后尺寸不变
        self.conv_tail = Conv_Block(num_stages*base_filter,base_filter*num_stages,3,1,1,'relu')
        self.conv_skip = Conv_Block(num_channels,num_stages*base_filter,3,1,1,'relu')
        self.last_conv = nn.Conv2d(base_filter//scale_factor ,num_channels,1,1,0)# 降维到channels=3,能够计算MSE loss

        #self.pixlshuffle = pix_shuffle(scale_factor)

        self.aaf_layer1 = aaf_layer(base_filter, kernel, stride, padding, num_stages=1)
        self.aaf_layer2 = aaf_layer(base_filter, kernel, stride, padding, num_stages=2)
        self.aaf_layer3 = aaf_layer(base_filter, kernel, stride, padding, num_stages=3)
        self.aaf_layer4 = aaf_layer(base_filter, kernel, stride, padding, num_stages=4)
        self.aff_layer4_conv4 = nn.Conv2d(base_filter,base_filter*num_stages,1,1,0)#提升维度用

        # 权重初始化
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
        x_copy = x
        x0 = self.head_layer(x)
        x1 = self.aaf_layer1(x0)
        concat_x01 = torch.cat((x1,x0),1)# 2,32,128,128
        x2 = self.aaf_layer2(concat_x01)
        concat_x012 = torch.cat((concat_x01,x2),1)

        x3 = self.aaf_layer3(concat_x012)
        concat_x0123 = torch.cat((concat_x012,x3),1)
        x4 = self.aaf_layer4(concat_x0123)
        x4 = self.aff_layer4_conv4(x4)
        out = self.conv_tail(x4)
        out = F.pixel_shuffle(out,upscale_factor = self.scale_factor)
        x_copy_conv = self.conv_skip(x_copy)
        x_copy_conv_pf = F.pixel_shuffle(x_copy_conv,upscale_factor= self.scale_factor)


        #x_copy_out = self.pixlshuffle()
        result = torch.add(x_copy_conv_pf,out)

        result = self.last_conv(result)

        return result

# if __name__ =="__main__":
#     net = AAF(num_channels=3,base_filter = 64,num_stages=4,scale_factor=2)
#     summary(net,(3,256,256))






