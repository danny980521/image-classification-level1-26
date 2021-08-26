import os,glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from albumentations import *
from albumentations.pytorch import ToTensorV2

import tqdm

# 테스트 데이터셋 폴더 경로를 지정해주세요.
test_dir = '/opt/ml/input/data/eval'

class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_paths)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet', type=str)
    parser.add_argument('--epoch_num', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str)

    args = parser.parse_args()
    
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    
    img_size = (512, 384)
    mean=(0.548, 0.504, 0.479)
    std=(0.237, 0.247, 0.246)
    transformations = {}
    transformations['test'] = Compose([
                                        Resize(img_size[0], img_size[1]),
                                        Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                                        ToTensorV2(p=1.0),
                                    ], p=1.0)
    
    dataset = TestDataset(image_paths)
    dataset.set_transform(transformations['test'])

    loader = DataLoader(
        dataset,
        shuffle=False
    )


    device = torch.device('cuda')
    
    age_model = torchvision.models.resnet18(pretrained=False)
    mask_model = torchvision.models.resnet18(pretrained=False)
    gender_model = torchvision.models.resnet18(pretrained=False)
    age_model.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
    mask_model.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
    gender_model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
    
    checkpoint_path = glob.glob('/opt/ml/checkpoint/{}*.pt'.format(args.model_path))[0]
    print('model checkpoint: ', checkpoint_path)
    
    checkpoint = torch.load(checkpoint_path)
    age_model.load_state_dict(checkpoint['age_model_state_dict'])
    mask_model.load_state_dict(checkpoint['mask_model_state_dict'])
    gender_model.load_state_dict(checkpoint['gender_model_state_dict'])

    age_model.eval()
    mask_model.eval()
    gender_model.eval()
    
    age_model = age_model.to(device)
    mask_model = mask_model.to(device)
    gender_model = gender_model.to(device)

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in tqdm.tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            
            age = age_model(images).argmax(dim=-1)
            mask = mask_model(images).argmax(dim=-1)
            gender = gender_model(images).argmax(dim=-1)
            
            pred = age + 6*mask + 3*gender
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')