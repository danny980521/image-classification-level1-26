import os,glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms
# from torchvision.transforms import Resize, ToTensor, Normalize
from albumentations import *
from albumentations.pytorch import ToTensorV2

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

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    
    # model_best_checkpoint = glob.glob('/opt/ml/checkpoint/{}_{}_*.pt'.format(args.model_name, args.epoch_num))[0]
    model_best_checkpoint = glob.glob('/opt/ml/checkpoint/{}*.pt'.format(args.model_path))[0]
    print('model checkpoint: ', model_best_checkpoint)
    model = torch.load(model_best_checkpoint)
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
    print('test inference is done!')