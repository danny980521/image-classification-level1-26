# Pstage_01_image_classification
Solution for Image Classification Competitions in 2nd BoostCamp AI Tech by **Round26 team**

## Content
- [Competition Abstract](#competition-abstract)
- [Hardware](#hardware)
- [Archive Contents](#archive-contents)
- [Getting Started](#getting-started)
	- [Dependencies](#dependencies)
	- [Install Requirements](#install-requirements)
	- [Training](#training)
	- [Inference](#inference)
	- [Make merge submission.csv File](#make-merge-submissioncsv-file)
- [Architecture](#architecture)



## Competition Abstract
- Classify age, gender and mask wearing from an input image

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Archive Contents
- image-classification-level1-26 : contains original code, trained models etc
```
image-classification-level1-26/
├── model/
│   ├── 5fold_mask/
│   ├── 5fold_gender/
│   └── age/
├── output/
│   ├── output_mask.csv
│   ├── output_gender.csv
│   └── output.csv
├── train_5fold.py
├── train_age.py
├── inference_5fold.py
├── inference_age.py
└── merger.py     
```
- `model/` : contains model to classify mask, gender and age
- `output/` : contains results from mask, gender, age model 
- `train_ .py` : trains age, mask, gender model
-  `inference_ .py` : inference age, mask, gender results
-  `merger.py` : merges inference csv files from inference_.py (final output)


## Getting Started    
### Dependencies
- torch==1.6.0
- torchvision==0.7.0
- tensorboard==2.4.1
- pandas==1.1.5
- scikit-learn~=0.24.1
- matplotlib==3.2.1
- timm==0.4.12
- wandb==0.12.1

### Install Requirements
```
pip install -r requirements.txt`
```

### Training
```
$ python train_5fold.py --task mask
$ python train_5fold.py --task gender
$ python train_age.py
```

### Inference
```
$ python inference_5fold.py --fold_dir ./model/5fold_mask --output_name output_mask
$ python inference_5fold.py --fold_dir ./model/5fold_gender --output_name output_gender
$ python inference_age.py --model_dir ./model/age
```

### Make merge submission.csv File
- Modify the result path in `merger.py` 
```
$ python merger.py
```



## Architecture
![](https://i.imgur.com/VmbxEYD.png)
