import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset.dataloader import getDataloader
import torchvision
from models.model import get_model
import argparse
from pytz import timezone
import datetime as dt

def train(args):
    now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%m%d_%H%M"))
    train_dataloader, val_dataloader = getDataloader(args)
    print('train_data: {}, val_data: {}'.format(len(train_dataloader), len(val_dataloader)))
    
    device = 'cuda'
    criterion = nn.CrossEntropyLoss()
    
    # torchvision models
    if args.model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)
    # elif args.model_name == 'vgg16':
    #     model = torchvision.models.vgg16(pretrained=False)
    #     model.classifier[6] = nn.Linear(in_features=4096, out_features=18, bias=True)
    
    # Custom models
    # if args.pretrained:
    #     model = torch.load(args.pretrained)
    # else:
    #     model = get_model(args)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    print(model)
    model = model.to(device)

    max_val_acc = 0
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_acc = 0
        print("----------------Train!----------------")
        for batch_iter, batch in enumerate(train_dataloader):
            batch_in, batch_out = batch
            batch_in = batch_in.to(device)
            batch_out = batch_out.to(device)

            # Forward path
            y_pred = model(batch_in)
            loss_out = criterion(y_pred, batch_out)
            acc_out = torch.sum(torch.max(y_pred, 1)[1] == batch_out.data)/len(batch_out.data)
            
            print('[epoch: {}, iter: {}] train acc: {:.5f}, train loss: {:.5f}'.format(epoch, batch_iter, acc_out, loss_out))
            optimizer.zero_grad()
            
            # backpropagate
            loss_out.backward()
            
            # optimizer update
            optimizer.step()
            total_loss += loss_out
            total_acc += acc_out
        avg_loss = total_loss/len(train_dataloader)
        avg_acc = total_acc/len(train_dataloader)
        print('[epoch: {}] average train acc: {:.5f}, average train loss: {:.5f}'.format(epoch, avg_acc, avg_loss))
        
        print("----------------validation!----------------")
        with torch.no_grad():
            model.eval() # evaluate (affects DropOut and BN)
            total_val_loss = 0
            total_val_acc = 0
            for val_iter, val_batch in enumerate(val_dataloader):
                val_batch_in, val_batch_out = val_batch
                val_batch_in, val_batch_out = val_batch_in.to(device), val_batch_out.to(device)
                y_pred = model(val_batch_in)
                loss_out = criterion(y_pred, val_batch_out)
                acc_out = torch.sum(torch.max(y_pred, 1)[1] == val_batch_out.data)/len(val_batch_out.data)
                print('[epoch: {}, iter: {}] validation acc: {}, validation loss: {}'.format(epoch, val_iter, acc_out, loss_out))
                total_val_loss += loss_out
                total_val_acc += acc_out
            avg_val_loss = total_val_loss/len(val_dataloader)
            avg_val_acc = total_val_acc/len(val_dataloader)
        
        # Print
        print ("epoch:[%d] val_acc: [%.5f], val_loss:[%.5f]."%(epoch, avg_val_acc, avg_val_loss))
        
        if avg_val_acc > max_val_acc:
            torch.save(model, '/opt/ml/checkpoint/{}_{}_{}_{:.5f}.pt'.format(now, args.model_name, epoch, avg_val_acc))
            max_val_acc = avg_val_acc
            print('----------------model saved!----------------')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--train_csv', default='/opt/ml/input/data/train/final_train.csv', type=str)
    parser.add_argument('--model_name', default='resnet', type=str)
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    
    train(args)