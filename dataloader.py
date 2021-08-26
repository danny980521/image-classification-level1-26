# import torch
# import pandas as pd
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from data import MyDataset
# from PIL import Image

# def getDataloader(args):
    
#     total_train_dataset = MyDataset(args, path = args.train_csv, 
#                                          transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
#                                                                          transforms.Resize((512, 512)), 
#                                                                          transforms.RandomHorizontalFlip(),
#                                                                          transforms.ToTensor(), 
#                                                                          transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.2,0.2,0.2)), ]), 
#                                          train=True)
#     train_len = int(args.train_ratio *len(total_train_dataset))
#     val_len = len(total_train_dataset) - train_len
    
#     train_dataset, val_dataset = torch.utils.data.random_split(total_train_dataset, lengths=[train_len, val_len])
    
#     train_loader = DataLoader(
#                             dataset=train_dataset,
#                             batch_size=args.batch_size, shuffle=True,
#                             num_workers=4, pin_memory=True)

#     val_loader = DataLoader(
#                             dataset=val_dataset,
#                             batch_size=args.batch_size, shuffle=False,
#                             num_workers=4, pin_memory=True)
    
#     return train_loader, val_loader



from sklearn.model_selection import train_test_split
from data import MaskBaseDataset
from augmentation import get_transforms

import torch
import torch.utils.data as data



def getDataloader(img_dir):
    mean, std = (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)
    transform = get_transforms(mean=mean, std=std)
    dataset = MaskBaseDataset(img_dir = img_dir)

    # train dataset과 validation dataset을 8:2 비율로 나눕니다.
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])
    
    train_dataset.dataset.set_transform(transform['train'])
    val_dataset.dataset.set_transform(transform['val'])

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=12,
        num_workers=4,
        shuffle=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=12,
        num_workers=4,
        shuffle=False
    )
    
    return train_loader, val_loader
