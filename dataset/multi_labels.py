import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2

class MultiLabels(Dataset):
    def __init__(self, args, path, train=True):
        self.args = args
        self.train = train
        self.data_csv = pd.read_csv(path)
        self.img_paths= self.data_csv['img_path']
        
        if self.train:
            self.labels = self.data_csv['class']
            self.genders = self.data_csv['gender']
            self.ages = self.data_csv['age']
            self.masks = self.data_csv['mask']
        else:
            self.labels, self.genders, self.ages, self.masks = None, None, None, None
    
    def set_transform(self, transform):
        self.transform = transform
        
    def __getitem__(self, ind):
        img = Image.open(self.img_paths[ind])
        if self.transform is not None:
            img = self.transform(image=np.array(img))['image']
        if not self.train:
            return img
        label = self.labels[ind]
        gender = self.genders[ind]
        age = self.ages[ind]
        mask = self.masks[ind]
        return img, label, gender, age, mask
    
    def __len__(self):
        return len(self.data_csv)