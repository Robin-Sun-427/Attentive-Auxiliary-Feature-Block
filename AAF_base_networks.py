import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self,in_size,out_size,ksize=3,stride=1,pad=1,activation=None):
        super(Conv_Block, self).__init__()

        self.conv = nn.Conv2d(in_size,out_size,ksize,stride,pad)
        self.activation = activation

        if self.activation is not None:
            self.act = nn.ReLU()

    def forward(self,x):
        out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class pix_shuffle(nn.Module):
    def __init__(self,activation='relu',up_scalefactor = 4):
        super(pix_shuffle, self).__init__()
        self.pixlshuffle = nn.PixelShuffle(up_scalefactor)
        self.activation = activation
        if self.activation is not None:
            self.act = nn.ReLU()

    def forward(self,x):
        out = self.pixlshuffle(x)
        if self.activation is not None:
            return self.act(out)
        return out


class aaf_layer(nn.Module):
    def __init__(self,num_filter,ksize=8,stride=4,padding=2,num_stages=1,
                                    activation='relu'):
        super(aaf_layer, self).__init__()

        self.conv1 = Conv_Block(num_filter*num_stages,num_filter,1,1,0,
                                activation=activation)
        self.averagepool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv2 = Conv_Block(num_filter,num_filter,1,1,0,
                                activation=activation)
        self.relu = nn.ReLU()
        self.conv3 = Conv_Block(num_filter,num_filter,1,1,0,
                                activation=activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.conv1(x)
        out = self.averagepool(out)
        out = self.relu(self.conv2(out))
        out = self.sigmoid(self.conv3(out))

        return out




