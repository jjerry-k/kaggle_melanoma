# %% 
import os, tqdm

import cv2 as cv
import numpy as np
import pandas as pd
import pydicom as di

from matplotlib import pyplot as plt

print("Package Loaded!")


# %%
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
ROOT = "./data"
TRAIN_IMG_DIR = os.path.join(ROOT, 'jpeg', 'train')
TEST_IMG_DIR = os.path.join(ROOT, 'jpeg', 'test')

tr_df = pd.read_csv('./data/train.csv')
te_df = pd.read_csv('./data/test.csv')

tr_names = list(tr_df['image_name'])
te_names = list(te_df['image_name'])

# %%
for i, val in tqdm.tqdm(enumerate(tr_names)):
    tr_names[i] = val + '.jpg'

for i, val in tqdm.tqdm(enumerate(te_names)):
    te_names[i] = val + '.jpg'


# %%
tr_df['image_name'] = tr_names
te_df['image_name'] = te_names

# %%
tr_df.to_csv("./data/new_train.csv")
te_df.to_csv("./data/new_test.csv")
