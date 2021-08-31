import os,glob
import pandas as pd
import numpy as np
import argparse
from PIL import Image
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    image_dir = os.path.join(test_dir, 'rmbg_images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    
    img_size = (512, 384)
    mean=(0.5, 0.5, 0.5)
    std=(0.2, 0.2, 0.2)
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
    #model_best_checkpoint = glob.glob('/opt/ml/checkpoint/{}*.pt'.format(args.model_path))[0]
    #print('model checkpoint: ', model_best_checkpoint)
    #model = torch.load(model_best_checkpoint)
    oof_pred = None
    fold0_dir = '/opt/ml/mybaseline/model/exp42/effb4_0_0_0.3697059926909528.pt'
    fold1_dir = '/opt/ml/mybaseline/model/exp42/effb4_1_1_0.365063661244135.pt'
    fold2_dir = '/opt/ml/mybaseline/model/exp42/effb4_2_1_0.46893066469287786.pt'
    fold3_dir = '/opt/ml/mybaseline/model/exp42/effb4_3_0_0.3057289303462328.pt'
    fold4_dir = '/opt/ml/mybaseline/model/exp42/effb4_4_1_0.4025938367737297.pt'
    model_dirs = [fold0_dir, fold1_dir, fold2_dir, fold3_dir, fold4_dir]
    for model_dir in model_dirs :
        model = torch.load(model_dir)
        model.eval()

        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        all_predictions = []
        with torch.no_grad():
            for images in tqdm.tqdm(loader):
                images = images.to(device)
                pred = model(images) / 2
                pred += model(images.transpose(Image.FL)) / 2
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        if oof_pred is None:
            oof_pred = fold_pred / 5
        else:
            oof_pred += fold_pred / 5
            
    submission['ans'] = np.argmax(oof_pred, axis=1)

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join('/opt/ml/submissions', 'submission_effb4_5fold_epoch2.csv'), index=False)
    print('test inference is done!')
