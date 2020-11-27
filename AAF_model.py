import torch
import torch.nn as nn
from torchsummary import summary

#=======================================================================================================================
class F_att_layer(nn.Module):#### F_att layer
    def __init__(self,lambda_mul_att=2):
        super(F_att_layer, self).__init__()
        self.average_pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.lambda_mul_att = lambda_mul_att
###
## x shape torch.Size([2, 3, 256, 256])
## out shape torch.Size([2, 64, 128, 128])
###
    def forward(self,x):
        x_copy = x
        #print("x shape",x.shape)
        out = self.average_pool(x)
        out = self.conv1(out)
        out = self.conv2(self.relu(out))
        #print("out shape",out.shape)
        out = self.sigmoid(out)
        #out = torch.matmul(out,x_copy) ####先注释 等开源
        out = out * self.lambda_mul_att
        return out

#=======================================================================================================================

class Projection_layer(nn.Module):#### projection_layer
    def __init__(self):
        super(Projection_layer, self).__init__()
        self.projection = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=1,
                                    stride=1,padding=0)

    def forward(self,x):
        out = self.projection(x)
        #print("out shape is",out.shape)
        return out
###
#   out shape is torch.Size([2, 64, 256, 256])
###

#=======================================================================================================================


class F_res_layer(nn.Module):
    def __init__(self,x_mutiply=1,lambda_mutiply=1):
        super(F_res_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.x_mutiply_value = x_mutiply
        self.lambda_value = lambda_mutiply

    def forward(self,x):
        x_copy = x * self.x_mutiply_value # x_multiply operation
        x_copy = self.conv1(x_copy)

        #print("x_copy ",x_copy.shape)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out * self.lambda_value # multiply operation
        #print("out shape ",out.shape)
        out = torch.add(out,x_copy)
        #print("result shape",out.shape)
        return out

"""
x_copy  torch.Size([2, 64, 256, 256])
out shape  torch.Size([2, 64, 256, 256])
result shape torch.Size([2, 64, 256, 256])
"""

#=======================================================================================================================

class skip_module(nn.Module):
    def __init__(self,up_scale_factor=2):
        super(skip_module, self).__init__()
        self.skip_conv_layer = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pixelshuffle  = nn.PixelShuffle(upscale_factor = up_scale_factor)

    def forward(self,x):
        #print("x shape",x.shape)
        out = self.skip_conv_layer(x)
        #print('out shape',out.shape)
        out = self.pixelshuffle(out)
        #print("out shape ",out.shape)
        return out
"""
x shape torch.Size([2, 3, 256, 256])
out shape torch.Size([2, 64, 256, 256])
out shape  torch.Size([2, 16, 512, 512])
"""

#=======================================================================================================================

class Tail_module(nn.Module):
    def __init__(self,up_scale_factor=2):
        super(Tail_module, self).__init__()
        self.tail_layer_conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pixelshuffle  = nn.PixelShuffle(upscale_factor = up_scale_factor)

    def forward(self,x):
        out = self.tail_layer_conv(x)
        #print("out shape ", out.shape)
        out = self.pixelshuffle(out)
        #print("out shape ", out.shape)
        return out

#=======================================================================================================================

class AAF_block_base(nn.Module):# No_linear mapping function
    def __init__(self,num_filter,num_stages=1):
        super(AAF_block_base, self).__init__()
        self.projection = Projection_layer(num_filter * num_stages,ksize,)
        self.f_att = F_att_layer()
        self.f_res = F_res_layer() #x_mutiply, lambda_mutiply

    def forward(self,x):
        out1 = self.projection(x)
        print(out1.shape)
        out1 = self.f_att(out1)
        print(out1.shape)
        out2 = self.f_res(x)
        print(out2.shape)
        out = torch.add(out1,out2)
        return out

#========================== 搭建AAF model ==================================#
class AAF_Model(nn.Module):
    def __init__(self):
        super(AAF_Model, self).__init__()
        self.head_layer = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.aaf = AAF_block_base()
        self.skip_layer = skip_module(up_scale_factor=1)
        self.tail_layer = Tail_module(up_scale_factor=1)

    def forward(self,x):
        skip_out = self.skip_layer(x) ## skip module

        x0 = self.head_layer(x)
        print("x0 shape is",x0.shape)

        x1 = self.aaf(x0)
        concat_feature = torch.cat((x0,x1),1)
        print("{}_{}_".format(x1.shape, concat_feature.shape))

        x2 = self.aaf(x1)
        concat_feature = torch.cat((concat_feature,x2),1)

        x3 = self.aff(x2)
        concat_feature = torch.cat((concat_feature,x3),1)

        x4 = self.aff(concat_feature)

        out = self.tail_layer(x4) ## Tail module
        out = torch.add(out,skip_out)
        return out

if __name__ =='__main__':

    net = AAF_block_base().to('cpu')
    summary(net,(64,256,256))



