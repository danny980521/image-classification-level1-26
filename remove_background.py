import os
import numpy as np
import torch
from enum import Enum
from PIL import Image
from PIL import ImageFile
from rembg.bg import remove
import io
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

_file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }


data_dir = '/opt/ml/input/data/train/images'
save_dir = '/opt/ml/input/data/train/rmbg_images'  # 배경 지운 이미지 저장할 폴더 지정

profiles = os.listdir(data_dir)
for profile in tqdm.tqdm(profiles):
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue

    img_folder = os.path.join(data_dir, profile)
    img_save_folder = os.path.join(save_dir, profile)
    if not os.path.exists(img_save_folder):  # profile마다 폴더 만들기
        os.makedirs(img_save_folder)

    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
        if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
            continue

        img_path = os.path.join(data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
        img = Image.open(io.BytesIO(remove(np.fromfile(img_path)))).convert("RGB")  # 배경 지우기
        img.save(os.path.join(img_save_folder, file_name))  # 배경 지운 이미지 저장
