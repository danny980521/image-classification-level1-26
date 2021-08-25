import torch
import pandas as pd
from torch.utils.data import DataLoader
# from torchvision import transforms
from albumentations import *
from albumentations.pytorch import ToTensorV2
from . import data, multi_labels

def getDataloader(args):
    
    img_size = (512, 384)
    mean=(0.548, 0.504, 0.479)
    std=(0.237, 0.247, 0.246)
    transformations = {}
    
    transformations['train'] = Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    
    transformations['val'] = Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    
    dataset = multi_labels.MultiLabels(args, args.train_csv, train=True)
    train_len = int(args.train_ratio*len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_dataset.dataset.set_transform(transformations['train'])
    val_dataset.dataset.set_transform(transformations['val'])
    
    train_loader = DataLoader(
                            dataset=train_dataset,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=4)

    val_loader = DataLoader(
                            dataset=val_dataset,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=4)

    return train_loader, val_loader