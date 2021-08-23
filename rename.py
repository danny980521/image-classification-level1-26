import os, glob

# 혹시 모르니까 복사해두고 새로운 곳에서 이름을 바꿔봅시다.(터미널에서 실행하세요)
# cp -r /opt/ml/input/data/train/images /opt/ml/input/data/train/new_images

new_path = '/opt/ml/input/data/train/new_images'
train_paths = glob.glob('/opt/ml/input/data/train/new_images/**/*.*')

for img_path in train_paths:
    info_list = img_path.split('/')[7].split('_')
    id, gender, race, age = info_list
    
    mask = os.path.basename(img_path)
    new_img_name = '_'.join([id, gender, race, age, mask])
    new_img_path = os.path.join('/'.join(img_path.split('/')[:-1]), new_img_name)
    
    os.rename(img_path, new_img_path)
    
    print('{} --> {}'.format(img_path, new_img_name))