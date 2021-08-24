import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset.dataloader import getDataloader
import torchvision
# from model.resnet import ResNet
import argparse

def train(args):
    train_dataloader, val_dataloader = getDataloader(args)
    print('train_data: {}, val_data: {}'.format(len(train_dataloader), len(val_dataloader)))
    
    device = 'cuda'
    criterion = nn.CrossEntropyLoss()
    
    if args.model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)
    elif args.model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=18, bias=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    print(model)
    model = model.to(device)

    min_val_loss = 999
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        print("----------------Train!----------------")
        for batch_iter, batch in enumerate(train_dataloader):
            batch_in, batch_out = batch

            # Forward path
            y_pred = model(batch_in.view(-1, 3, 512, 384).to(device).float())
            loss_out = criterion(y_pred, torch.max(batch_out, 1)[1].to(device))
            print('[epoch: {}, iter: {}] train loss: {}'.format(epoch, batch_iter, loss_out))
            optimizer.zero_grad()
            
            # backpropagate
            loss_out.backward()
            
            # optimizer update
            optimizer.step()
            total_loss += loss_out
        avg_loss = total_loss/len(train_dataloader)
        print('[epoch: {}] average train loss: {}'.format(epoch, avg_loss))
        
        print("----------------validation!----------------")
        with torch.no_grad():
            model.eval() # evaluate (affects DropOut and BN)
            total_val_loss = 0
            for val_iter, val_batch in enumerate(val_dataloader):
                val_batch_in, val_batch_out = val_batch
                y_pred = model(val_batch_in.view(-1, 3, 512, 384).to(device).float())
                loss_out = criterion(y_pred, torch.max(val_batch_out, 1)[1].to(device))
                print('[epoch: {}, iter: {}] validation loss: {}'.format(epoch, val_iter, loss_out))
                total_val_loss += loss_out
            avg_val_loss = total_val_loss/len(val_dataloader)
        
        # Print
        print ("epoch:[%d] val_loss:[%.3f]."%(epoch, avg_val_loss))
        
        if avg_val_loss < min_val_loss:
            torch.save(model, '/opt/ml/checkpoint/{}_{}_{}.pt'.format(args.model_name, epoch, avg_val_loss))
            min_val_loss = avg_val_loss
            print('----------------model saved!----------------')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--train_csv', default='/opt/ml/input/data/train/train_label.csv', type=str)
    parser.add_argument('--model_name', default='resnet', type=str)
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    
    train(args)