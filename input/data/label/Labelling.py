import os
import sys
import numpy as np
import pandas as pd
from glob import glob

path = '../input/data/train'
img_dir = '../input/data/train/images'

def make_img_path():
    df = pd.read_csv(os.path.join(path, 'train_update.csv'))
    images = []
    images_path = [os.path.join(img_dir, img_file) for img_file in df.path.values]
    for img_path in images_path:
        for image in glob(os.path.join(img_path, '*')):
            images.append(image)
    return images

def labeling(gender, age, mask_id):
    age_code = [0, 1, 2]
    gender_code = [0, 3]
    mask_code = [0, 6, 12]
    age_idx, gender_idx, mask_idx = 0, 0, 0
    if age < 30 :
        age_idx = 0
    elif 30 <= age < 60 :
        age_idx = 1
    else :
        age_idx = 2
    if gender == 'male' :
        gender_idx = 0
    else :
        gender_idx = 1
    if mask_id[0] == 'm' :
        mask_idx = 0
    elif mask_id[0] == 'i' :
        mask_idx = 1
    elif mask_id[0] == 'n' :
        mask_idx = 2
    return dict(age_code = age_code[age_idx],gender_code = gender_code[gender_idx], mask_code = mask_code[mask_idx])
        

def make_label(images_path):
    label = []
    for image in images_path:
        person_id, gender, race, age = image.split('/')[-2].split('_')
        mask_id = os.path.splitext(image.split('/')[-1])[0]
        label.append(labeling(gender, int(age), mask_id))
    return label

def make_new_train(images_path, label):
    csv_list = []
    csv_column = ['path', 'age_code', 'gender_code', 'mask_code', 'class']
    for path, codes in zip(images_path, label):
        real_image_path = os.path.join(path.split('/')[-2], path.split('/')[-1])
        temp = [real_image_path] + list(codes.values())
        temp.append(sum(codes.values()))
        csv_list.append(temp)
    df = pd.DataFrame(data = csv_list, columns = csv_column)
    df.to_csv('../input/data/train/new_train.csv', index = False)    


if __name__ == '__main__':
    images_path = make_img_path()
    label = make_label(images_path)
    make_new_train(images_path, label)