import pandas as pd

dir = '/opt/ml/'

mask_path = f'{dir}/submissions/submission_effb4mask_5fold_epoch2.csv' # 마스크 분류 결과 csv 파일 경로를 입력하세요
gender_path = f'{dir}/submissions/submission_effb4gender_5fold_epoch2.csv'  # 성별 분류 결과 csv 파일 경로를 입력하세요
age_path = f'{dir}/submissions/best.csv'  # 나이 분류 결과 csv 파일 경로를 입력하세요

mask_df = pd.read_csv(mask_path)
gender_df = pd.read_csv(gender_path)
age_df = pd.read_csv(age_path)

mask_df['ans'] = mask_df['ans'] * 6 + gender_df['ans'] * 3 + age_df['ans']

mask_df.to_csv(f'{dir}/submissions/merged_ensemble_5fold.csv')