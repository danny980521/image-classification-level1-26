# import torch
# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image

# class MyDataset(Dataset):
#     def __init__(self, args, path, transform, train = True):
#         self.args = args
#         self.transform = transform
#         self.train = train
#         self.df_csv = pd.read_csv(path)
#         self.img_path = self.df_csv['img_path']
#         self.labels = self.df_csv['class'] if self.train else None
        
#     def __getitem(self, index):
#         img = Image.open(self.img_path[index])
        
#         if self.transform is not None:
#             img = self.transform(img)
        
#         if not self.train:
#             return img
        
#         label_tensor = torch.tensor(np.eye(18)[labels[150]], dtype=torch.long) # one hot encoding
        
#         return img, label_tensor
    
#     def __len__(self):
#         return len(self.df_csv)
import os
import numpy as np
from PIL import Image
from labels import MaskLabels, GenderLabels, AgeGroup

import torch
import torch.utils.data as data


class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3
    
    _file_names = {
        "mask1.jpg": MaskLabels.mask,
        "mask2.jpg": MaskLabels.mask,
        "mask3.jpg": MaskLabels.mask,
        "mask4.jpg": MaskLabels.mask,
        "mask5.jpg": MaskLabels.mask,
        "incorrect_mask.jpg": MaskLabels.incorrect,
        "normal.jpg": MaskLabels.normal
    }
    
    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    
    def __init__(self, img_dir, transform=None):
        """
        MaskBaseDataset을 initialize 합니다.

        Args:
            img_dir: 학습 이미지 폴더의 root directory 입니다.
            transform: Augmentation을 하는 함수입니다.
        """
        self.img_dir = img_dir
        self.transform = transform
        
        self.setup()
        
    def set_transform(self, transform):
        """
        transform 함수를 설정하는 함수입니다.
        """
        self.transform = transform
    
        
    def setup(self):
        """
        image의 경로와 각 이미지들의 label을 계산하여 저장해두는 함수입니다.
        여기는 csv 파일을 따로 안만들고 리스트에 저장함 
        """
        profiles = os.listdir(self.img_dir)
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_dir, profile, file_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_labels.append(label)

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(GenderLabels, gender)
                    age_label = AgeGroup.map_label(age)

                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    
    def __getitem__(self, index):
        """
        데이터를 불러오는 함수입니다. 
        데이터셋 class에 데이터 정보가 저장되어 있고, index를 통해 해당 위치에 있는 데이터 정보를 불러옵니다.
        
        Args:
            index: 불러올 데이터의 인덱스값입니다.
        """
        # 이미지를 불러온다
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        # 레이블을 불러온다
        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        multi_class_label = mask_label * 6 + gender_label * 3 + age_label
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, multi_class_label
        
    def __len__(self):
        return len(self.image_paths)