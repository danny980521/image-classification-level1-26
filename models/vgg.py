import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.autograd import Variable
import torch.utils.data as data
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        #nn.BatchNorm1d(size_out),
        nn.ReLU(),
        nn.Dropout(0.5),
    )
    return layer

class VGG16(nn.Module):
    def __init__(self, n_classes=18):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,32], [32,32], [3,3], [1,1], 2, 2) #512->256 , 384->192
        self.layer2 = vgg_conv_block([32,64], [64,64], [3,3], [1,1], 2, 2) #256->128, 192->96
        self.layer3 = vgg_conv_block([64,64], [64,64], [3,3], [1,1], 2, 2) #128->64 ,96->48
        #self.layer4 = vgg_conv_block([128,512], [512,512], [3,3], [1,1], 2, 2) #100->100 pooling->25
        #self.layer4 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2) #50->50->50 pooling->25
        #self.layer5 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2) #25->25->25 pooling->12

        # FC layers
        #self.layer6 = vgg_fc_layer(50*50*64, 4096)
        self.layer6 = vgg_fc_layer(64*48*64, 4096)
        #self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        vgg16_features = self.layer3(out)
        #vgg16_features = self.layer4(out)
        
        #vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size()[0],-1)
        out = self.layer6(out)
        #out = self.layer7(out)
        out = self.layer8(out)

        return out