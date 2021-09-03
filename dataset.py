import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from pandas_streaming.df import train_test_apart_stratify
from torchvision import transforms
from torchvision.transforms import *

import albumentations as albu
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

    
class AlbuAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = albu.Compose([
            albu.Resize(resize[0], resize[1]),
            albu.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    def __call__(self, image):
        return self.transform(image=np.array(image))


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2, transform = None):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = transform
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
    
    
    
class ElevenAgeLabels(int, Enum):
    '''
    나이 클래스를 11개로 세분화하는 클래스
    '''
    Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if 0 < value <= 18:
            return cls.Zero
        elif value == 19:
            return cls.One
        elif value == 20:
            return cls.Two
        elif 21 <= value <= 24:
            return cls.Three
        elif 25 <= value <= 29:
            return cls.Four
        elif 30 <= value <= 48:
            return cls.Five
        elif 49 <= value <= 52:
            return cls.Six
        elif 53 <= value <= 55:
            return cls.Seven
        elif 56 <= value <= 57:
            return cls.Eight
        elif 58 <= value <= 59:
            return cls.Nine
        elif 60 <= value :
            return cls.Ten


class AgeBaseDataset(MaskBaseDataset):
    num_classes = 11

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }
    
    all_labels = []
    indexs = []
    groups = []
    original_age_labels = []

    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        super().__init__(data_dir)

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                org_age_label = AgeLabels.from_number(age)
                age_label = ElevenAgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.original_age_labels.append(org_age_label)
                self.age_labels.append(age_label)
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))
                self.indexs.append(cnt)
                self.groups.append(id)
                cnt += 1

    def get_age_label(self, index):
        return self.original_age_labels[index], self.age_labels[index]

    @staticmethod
    def encode_original_age(age_label) -> int:
        original_age_label = torch.zeros_like(age_label)
        for ind, age in enumerate(age_label):
            if age <= 4:
                original_age_label[ind] = 0
            elif age < 10:
                original_age_label[ind] = 1
            else:
                original_age_label[ind] = 2
        return original_age_label

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        org_age_label, age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, org_age_label.value, age_label.value
    
    def split_dataset(self) -> Tuple[Subset, Subset]:
        df = pd.DataFrame({"indexs":self.indexs, "groups":self.groups, "labels":self.age_labels})
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()
        
        return [Subset(self, train_index), Subset(self, valid_index)]


class My5foldDataset(MaskBaseDataset):
  
    '''
    K-Fold를 하기 위한 데이터 세트 
    '''
    all_labels = [] 
    indexs = [] 
    groups = [] 
    
    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def setup(self):
        cnt = 0 
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label)) 
                self.indexs.append(cnt) 
                self.groups.append(id) 
                cnt += 1 

    def split_dataset(self) -> Tuple[Subset, Subset]:
        df = pd.DataFrame({"indexs":self.indexs, "groups":self.groups, "labels":self.all_labels})

        train, valid1 = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train, valid2 = train_test_apart_stratify(train, group="groups", stratify="labels", test_size=0.25)
        train, valid3 = train_test_apart_stratify(train, group="groups", stratify="labels", test_size=1/3)
        valid4, valid5 = train_test_apart_stratify(train, group="groups", stratify="labels", test_size=0.5)
        valid1_index = valid1["indexs"].tolist()
        valid2_index = valid2["indexs"].tolist()
        valid3_index = valid3["indexs"].tolist()
        valid4_index = valid4["indexs"].tolist()
        valid5_index = valid5["indexs"].tolist()
        train1_index = valid2_index + valid3_index + valid4_index + valid5_index
        train2_index = valid1_index + valid3_index + valid4_index + valid5_index
        train3_index = valid1_index + valid2_index + valid4_index + valid5_index
        train4_index = valid1_index + valid2_index + valid3_index + valid5_index
        train5_index = valid1_index + valid2_index + valid3_index + valid4_index

        return [[Subset(self, train1_index), Subset(self, valid1_index)],[Subset(self, train2_index), Subset(self, valid2_index)],[Subset(self, train3_index), Subset(self, valid3_index)],[Subset(self, train4_index), Subset(self, valid4_index)],[Subset(self, train5_index), Subset(self, valid5_index)]]
    
