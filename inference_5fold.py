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

from dataset import TestDataset


# 테스트 데이터셋 폴더 경로를 지정해주세요.
test_dir = '/opt/ml/input/data/eval'

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def inference(args):
    
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    
    dataset = TestDataset(image_paths, args.resize)
    
    loader = DataLoader(
        dataset,
        shuffle=False
    )

    device = torch.device('cuda')
    
    oof_pred = None
    
    # get 5 best models
    best_model_paths = []
    for fold_num in range(5):
        model_f1_score = []

        model_paths = glob.glob(os.path.join(args.fold_dir, f'effb4_{fold_num}_*.pt'))

        for model_path in model_paths:
            model_f1_score.append(float(os.path.basename(model_path).split('_')[3]))

        best_model_paths.append(model_paths[torch.argmax(torch.FloatTensor(model_f1_score))])
    
    model_dirs = best_model_paths
    print(model_dirs)
    
    for model_dir in model_dirs :
        model = torch.load(model_dir)
        model.eval()

        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        all_predictions = []
        with torch.no_grad():
            for images in tqdm.tqdm(loader):
                images = images.to(device)
                pred = model(images)
                all_predictions.extend(pred.cpu().numpy())

            fold_pred = np.array(all_predictions)

        if oof_pred is None:
            oof_pred = fold_pred / 5
        else:
            oof_pred += fold_pred / 5
            
    submission['ans'] = np.argmax(oof_pred, axis=1)

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(args.output_path, args.output_name+'.csv'), index=False)
    print('test inference is done!')
    
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (defult:32)')
    parser.add_argument('--model', type=str, default='EfficientnetB4', help='model type (default: EfficientnetB4)')
    parser.add_argument('--resize', nargs='+', type=int, default=[256,192], help='resize size for image when inference(default: (256,192))')
    parser.add_argument('--output_path', default='./output', type=str)
    parser.add_argument('--fold_dir', default='./model', type=str)
    parser.add_argument('--output_name', default='output', type=str)

    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    inference(args)
    
    
    