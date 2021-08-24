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
from models.resnet import ResNet, resnet18, resnet34
from models.vgg import VGG16

def get_model(args):
    if args.model_name.find('resnet') != -1:
        if args.model_name.find('18') != -1:
            model = resnet18()
        elif args.model_name.find('34') != -1:
            model = resnet34()
    elif args.model_name.find('vgg') != -1:
        model = VGG16()
    return model