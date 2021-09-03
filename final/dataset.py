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
#from torchvision import transforms
#from torchvision.transforms import *

from albumentations import *
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


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)
    
class TrainAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize[0], resize[1], p=1.0),
            #HorizontalFlip(p=0.5),
            #ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.0, sat_shift_limit=0.5, val_shift_limit=0.5, p=1),
            #RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            #GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

    def __call__(self, image):
        return self.transform(image=np.array(image))
    
    
class ValidAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(resize[0], resize[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            HueSaturationValue(hue_shift_limit=0.0, sat_shift_limit=0.5, val_shift_limit=0.5, p=1),
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
        return image_transform, age_label

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
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
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

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


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
    
class MyMaskDataset(MaskBaseDataset):

    all_labels = [] # 추가
    indexs = [] # 추가
    groups = [] # 추가
    
    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def setup(self):
        cnt = 0 # 추가
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
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label)) # 추가
                self.indexs.append(cnt) # 추가
                self.groups.append(id) # 추가
                cnt += 1 # 추가

    def split_dataset(self) -> Tuple[Subset, Subset]:
        df = pd.DataFrame({"indexs":self.indexs, "groups":self.groups, "labels":self.all_labels})

        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()

        return [Subset(self, train_index), Subset(self, valid_index)]
    
class AgeLabels_2(int, Enum):
    Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Eleven = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

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
            return cls.Seven
        elif 53 <= value <= 55:
            return cls.Eight
        elif 56 <= value <= 57:
            return cls.Nine
        elif 58 <= value <= 59:
            return cls.Ten
        elif 60 <= value :
            return cls.Eleven


class AgeBaseDataset(MaskBaseDataset):
    num_classes = 12

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }
    
    all_labels = [] # 추가
    indexs = [] # 추가
    groups = [] # 추가
    original_age_labels = []

    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        # self.transform = transform
        super().__init__(data_dir)

    def setup(self):
        cnt = 0 # 추가
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
                age_label = AgeLabels_2.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.original_age_labels.append(org_age_label)
                self.age_labels.append(age_label)
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label)) # 추가
                self.indexs.append(cnt) # 추가
                self.groups.append(id) # 추가
                cnt += 1 # 추가

    def get_age_label(self, index):
        return self.original_age_labels[index], self.age_labels[index]

    @staticmethod
    def encode_original_age(age_label) -> int:
        original_age_label = torch.zeros_like(age_label)
        for ind, age in enumerate(age_label):
            if age <= 4:
                original_age_label[ind] = 0
            elif age <= 10:
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

class My5foldDataset(MaskBaseDataset):

    all_labels = [] # 추가
    indexs = [] # 추가
    groups = [] # 추가
    
    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def setup(self):
        cnt = 0 # 추가
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
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label)) # 추가
                self.indexs.append(cnt) # 추가
                self.groups.append(id) # 추가
                cnt += 1 # 추가

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
    
class alltrainDataset(MaskBaseDataset):

    all_labels = [] # 추가
    indexs = [] # 추가
    groups = [] # 추가
    
    def __init__(self, data_dir, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2), val_ratio=0.2):
        super().__init__(data_dir, mean, std, val_ratio)
        
    def setup(self):
        cnt = 0 # 추가
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
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label)) # 추가
                self.indexs.append(cnt) # 추가
                self.groups.append(id) # 추가
                cnt += 1 # 추가

    def split_dataset(self) -> Tuple[Subset, Subset]:
        df = pd.DataFrame({"indexs":self.indexs, "groups":self.groups, "labels":self.all_labels})

        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=0)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()

        return [Subset(self, train_index), Subset(self, valid_index)]
    