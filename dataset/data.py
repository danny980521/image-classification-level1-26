import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2

class Data(Dataset):
    def __init__(self, args, path, train=True):
        self.args = args
        self.train = train
        self.data_csv = pd.read_csv(path)
        self.img_paths= self.data_csv['img_path']
        self.labels = self.data_csv['class'] if self.train else None
    
    def set_transform(self, transform):
        self.transform = transform
        
    def __getitem__(self, ind):
        img = Image.open(self.img_paths[ind])
        if self.transform is not None:
            img = self.transform(image=np.array(img))['image']
        if not self.train:
            return img
        label = self.labels[ind]
        return img, label
        '''
        label_tensor = torch.zeros(len(self.labels.unique()), dtype=torch.long) if self.train else None
        label_ind = self.labels[ind]
        label_tensor[label_ind] = 1.
        # label = torch.tensor(self.labels, dtype=torch.long)
        return img, label_tensor
        '''
        
    
    def __len__(self):
        return len(self.data_csv)