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

data_dir = '/opt/ml/input/data/eval/images'
save_dir = '/opt/ml/input/data/eval/rmbg_images'  # 배경 지운 이미지 저장할 폴더 지정

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image_file = os.listdir(data_dir)
for image in tqdm.tqdm(image_file):
    if image.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue

    img_path = os.path.join(data_dir, image)
    img_save_path = os.path.join(save_dir, image)

    img = Image.open(io.BytesIO(remove(np.fromfile(img_path)))).convert("RGB")  # 배경 지우기
    img.save(img_save_path)  # 배경 지운 이미지 저장
