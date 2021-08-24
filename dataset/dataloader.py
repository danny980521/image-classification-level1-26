import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from . import data

def getDataloader(args):
    
    total_train_dataset = data.Data(args, args.train_csv,
                            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)), ]),
                                                        train=True)
    train_len = int(args.train_ratio*len(total_train_dataset))
    val_len = len(total_train_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(total_train_dataset, lengths=[train_len, val_len])
    
    train_loader = DataLoader(
                            dataset=train_dataset,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    val_loader = DataLoader(
                            dataset=val_dataset,
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    return train_loader, val_loader