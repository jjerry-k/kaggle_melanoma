# %%
import os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import wandb
from wandb.keras import WandbCallback
print("Package Loaded!")
# %%
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# %%
ROOT = './data'

TR_CSV_PATH = os.path.join(ROOT, "new_train.csv")
TE_CSV_PATH = os.path.join(ROOT, "new_test.csv")

TR_IMG_PATH = os.path.join(ROOT, 'jpeg', 'train')
TE_IMG_PATH = os.path.join(ROOT, 'jpeg', 'test')

SIZE = 224
BATCH_SIZE = 128
SEED = 777
EPOCHS=10

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

date = time.ctime()[:-14].replace(' ', '_')
wandb.init(project="kaggle_melanoma", name=date)
# %%
tr_csv = pd.read_csv(TR_CSV_PATH)
tr_csv['target']= tr_csv['target'].astype("str")
tr_csv, val_csv = train_test_split(tr_csv, test_size = 0.05, random_state=SEED, shuffle=True)
print(tr_csv.head(5))
print(val_csv.head(5))

te_csv = pd.read_csv(TE_CSV_PATH)
print(te_csv.head(5))

# %%
from tensorflow.keras import layers, models, losses, optimizers, callbacks
from tensorflow.keras.applications import Xception
base_model = Xception(weights="imagenet", include_top=False, pooling='avg')
base_model.summary()

out = layers.Dense(1, activation="sigmoid")(base_model.output)

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
val_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


train_generator = train_image_generator.flow_from_dataframe(tr_csv, directory=TR_IMG_PATH, \
                                                                x_col='image_name', y_col='target',\
                                                                target_size=(SIZE, SIZE), class_mode='binary',\
                                                                batch_size=BATCH_SIZE, seed=SEED)

val_generator = val_image_generator.flow_from_dataframe(val_csv, directory=TR_IMG_PATH, \
								x_col='image_name', y_col='target',\
                                                                target_size=(SIZE, SIZE), class_mode='binary',\
                                                                batch_size=BATCH_SIZE, seed=SEED)

test_generator = test_image_generator.flow_from_dataframe(te_csv, directory=TE_IMG_PATH,\
                                                                x_col='image_name', y_col=None,\
                                                                class_mode=None, target_size=(SIZE, SIZE),\
                                                                batch_size=BATCH_SIZE, seed=SEED)

reducelr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=4, verbose=1)
earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
wandb_callback = WandbCallback(input_type='image', labels=[0,1], save_weights_only=True)

# %%
model = models.Model(base_model.input, out)
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# %%
model.fit(train_generator, \
            epochs=EPOCHS, \
            validation_data=val_generator, \
            callbacks = [reducelr, earlystop, wandb_callback], \
            workers=4)

# %%
