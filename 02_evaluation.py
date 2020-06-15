# %%
import os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# %%
ROOT = './data'

TE_CSV_PATH = os.path.join(ROOT, "new_test.csv")

TE_IMG_PATH = os.path.join(ROOT, 'jpeg', 'test')

SIZE = 224
BATCH_SIZE = 16
SEED = 777
EPOCHS=10

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%
te_csv = pd.read_csv(TE_CSV_PATH)
print(te_csv.head(5))

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

test_generator = test_image_generator.flow_from_dataframe(te_csv, directory=TE_IMG_PATH,\
                                                                x_col='image_name', y_col=None,\
                                                                class_mode=None, target_size=(SIZE, SIZE),\
                                                                batch_size=BATCH_SIZE, seed=SEED)


# %%
from tensorflow.keras import layers, models, losses, optimizers, callbacks
from tensorflow.keras.applications import Xception
base_model = Xception(weights="imagenet", include_top=False, pooling='avg')
base_model.summary()

out = layers.Dense(1, activation="sigmoid")(base_model.output)
model = models.Model(base_model.input, out)
model.load_weights('/home/ubuntu/kaggle/wandb/run-20200611_131020-2zx2z087/model-best.h5')

# %%
result = model.predict(test_generator, verbose=1)
# %%
SP_CSV_PATH = os.path.join(ROOT, "sample_submission.csv")
sample_csv = pd.read_csv(SP_CSV_PATH)
sample_csv['target'] = result
sample_csv.to_csv("./submission.csv", index=False)

