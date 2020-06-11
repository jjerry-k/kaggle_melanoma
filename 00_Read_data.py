# %% 
import os

import cv2 as cv
import numpy as np
import pandas as pd
import pydicom as di

from matplotlib import pyplot as plt

print("Package Loaded!")

# %%
ROOT = "./data"

JPG_DIR = os.path.join(ROOT, "jpeg", "train")

DCM_DIR = os.path.join(ROOT, "train")

# %%
jpg_list = os.listdir(JPG_DIR)

dcm_list = os.listdir(DCM_DIR)

# %%
print(jpg_list[:10])

print(dcm_list[:10])

# %%
jpg_img = cv.imread(os.path.join(JPG_DIR, jpg_list[0]))
jpg_img = cv.cvtColor(jpg_img, cv.COLOR_BGR2RGB)

dcm_img = di.dcmread(os.path.join(DCM_DIR, dcm_list[0])).pixel_array

# %%
# Two image are different
plt.subplot(121)
plt.imshow(jpg_img)
plt.subplot(122)
plt.imshow(dcm_img)
# %%
