import pandas as pd
import os, glob

# csv_file = pd.read_csv('/opt/ml/input/data/train/tain.csv')
def make_class(gender, age, mask):
    age = int(age)
    if age < 30:
        if gender == 'male':
            if mask.find('incorrect') != -1: label = 6
            elif mask == 'normal': label = 12
            else: label = 0
        else: # female
            if mask.find('incorrect') != -1: label = 9
            elif mask == 'normal': label = 15
            else: label = 3
    elif 30 <= age < 60:
        if gender == 'male':
            if mask.find('incorrect') != -1: label = 7
            elif mask == 'normal': label = 13
            else: label = 1
        else: # female
            if mask.find('incorrect') != -1: label = 10
            elif mask == 'normal': label = 16
            else: label = 4
    else: # over 60
        if gender == 'male':
            if mask.find('incorrect') != -1: label = 8
            elif mask == 'normal': label = 14
            else: label = 2
        else:
            if mask.find('incorrect') != -1: label = 11
            elif mask == 'normal': label = 17
            else: label = 5
    return label

train_paths = glob.glob('/opt/ml/input/data/train/new_images/**/*.*')

train_df = pd.DataFrame(columns=('id', 'gender', 'race', 'age', 'mask', 'img_path', 'class'))

for ind, img_path in enumerate(train_paths):
    print('processing.. {}'.format(img_path))
    split_list = os.path.basename(img_path).split('_')
    
    if len(split_list) == 5:
        id, gender, race, age, mask = split_list
    else:
        id, gender, race, age, incorrect, mask = split_list
        mask = '_'.join([incorrect, mask])
    
    label = make_class(gender, age, mask)
    train_df.loc[ind] = [id, gender, race, age, mask, img_path, label]

train_df.to_csv('/opt/ml/input/data/train/train_label.csv')