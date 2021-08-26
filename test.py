import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

from data import MaskBaseDataset
from dataloader import getDataloader

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from adamp import AdamP


data_dir = '/opt/ml/input/data/train'
img_dir = f'{data_dir}/images'
df_path = f'{data_dir}/train2.csv'
df = pd.read_csv(df_path)  

train_dataloader, val_dataloader = getDataloader(img_dir)
print('train_data: {}, val_data: {}'.format(len(train_dataloader), len(val_dataloader)))

device = 'cuda'

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 18)

print(model)
model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = AdamP(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-2)

min_val_loss = 999
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    print("----------------Train!----------------")
    for i, (images, labels) in enumerate(tqdm(train_dataloader)):
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss_out = loss_function(outputs, labels)
        #print('[epoch: {}, iter: {}] train loss: {}'.format(epoch, i, loss_out))
        
        loss_out.backward()
        optimizer.step()
        total_loss += loss_out
        
    avg_loss = total_loss/len(train_dataloader)
    print('[epoch: {}] average train loss: {}'.format(epoch, avg_loss))
    
    print("----------------validation!----------------")
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        for i, (images,labels) in enumerate(tqdm(val_dataloader)):
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss_out = loss_function(outputs, labels)
            #print('[epoch: {}, iter: {}] validation loss: {}'.format(epoch, val_iter, loss_out))
            
            total_val_loss += loss_out
        avg_val_loss = total_val_loss / len(val_dataloader)
        
    print ("epoch:[%d] val_loss:[%.3f]."%(epoch, avg_val_loss))
    
    if avg_val_loss < min_val_loss:
        torch.save(model, '/opt/ml/checkpoint/{}_{}_{}.pt'.format('resnet18_adamp', epoch, avg_val_loss))
        min_val_loss = avg_val_loss
        print('----------------model saved!----------------')
            
        
    



