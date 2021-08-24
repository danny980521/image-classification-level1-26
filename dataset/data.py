import torch
import pandas as pd
from torch.utils.data import Dataset
import imageio

class Data(Dataset):
    def __init__(self, args, path, transform, train=True):
        self.args = args
        self.transform = transform
        self.train = train
        self.data_csv = pd.read_csv(path)
        self.img_paths= self.data_csv['img_path']
        self.labels = self.data_csv['class'] if self.train else None
        
    def __getitem__(self, ind):
        img = torch.tensor(imageio.imread(self.img_paths[ind]), dtype=float)
        if self.transform is not None:
            img = self.transform(img)
        if not self.train:
            return img
        label_tensor = torch.zeros(len(self.labels.unique()), dtype=torch.long) if self.train else None
        label_ind = self.labels[ind]
        label_tensor[label_ind] = 1.
#         label = torch.tensor(self.labels, dtype=torch.long)
        return img, label_tensor
    
    def __len__(self):
        return len(self.data_csv)