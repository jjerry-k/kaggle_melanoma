# %%
import os, random, time, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, losses, optimizers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3
from tensorflow.keras.applications import Xception, MobileNet, MobileNetV2, ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbCallback
print("Package Loaded!")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. {['yes', 'true', 't', 'y', '1'], ['no', 'false', 'f', 'n', '0']}")

MODEL_LIST = {
    'vgg16': VGG16, 
    'vgg19': VGG19, 
    'resnet50': ResNet50, 
    'resnet101': ResNet101, 
    'resnet152': ResNet152, 
    'inceptionv3': InceptionV3, 
    'xception': Xception, 
    'mobilenet': MobileNet, 
    'mobilenetv2': MobileNetV2, 
    'resnet50v2': ResNet50V2, 
    'resnet101v2': ResNet101V2, 
    'resnet152v2': ResNet152V2, 
    "inceptionresnetv2": InceptionResNetV2, 
    'densenet121': DenseNet121, 
    'densenet169': DenseNet169, 
    'densenet201': DenseNet201,
    'efficientnetb0': EfficientNetB0,
    'efficientnetb1': EfficientNetB1,
    'efficientnetb2': EfficientNetB2,
    'efficientnetb3': EfficientNetB3,
    'efficientnetb4': EfficientNetB4,
    'efficientnetb5': EfficientNetB5,
    'efficientnetb6': EfficientNetB6,
    'efficientnetb7': EfficientNetB7,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_number", type=str, help="GPU number to use")
parser.add_argument("--model", type=str, help="Pretrained model to use")
parser.add_argument("--freeze", type=str2bool, nargs='?', const=True, default=False,help="Freezing pretrained model")
parser.add_argument("--epochs", type=int, help="Number of epochs")

args = parser.parse_args()
GPU_NUM = args.gpu_number
EPOCHS = args.epochs
FREEZE = args.freeze
MODEL = args.model.lower()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM

# %%
# For Efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# %%
ROOT = '/data/jerry/private/data'

TR_CSV_PATH = os.path.join(ROOT, "new_train.csv")
TE_CSV_PATH = os.path.join(ROOT, "new_test.csv")

TR_IMG_PATH = os.path.join(ROOT, 'jpeg', 'train')
TE_IMG_PATH = os.path.join(ROOT, 'jpeg', 'test')

SIZE = 224
BATCH_SIZE = 1024
SEED = 777

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

date = time.ctime()[:-14].replace(' ', '_')
wandb.init(project="kaggle_melanoma", name=f"{date}_{MODEL}_{FREEZE}")
# %%
tr_csv = pd.read_csv(TR_CSV_PATH)
tr_csv['target']= tr_csv['target'].astype("str")
tr_csv, val_csv = train_test_split(tr_csv, test_size = 0.05, random_state=SEED, shuffle=True)

df_1=tr_csv[tr_csv['target']=='1']
df_0=tr_csv[tr_csv['target']=='0'].sample(len(df_1))
tr_csv=pd.concat([df_0, df_1])
tr_csv=tr_csv.reset_index()

print(tr_csv.head(5))
print(val_csv.head(5))

te_csv = pd.read_csv(TE_CSV_PATH)
print(te_csv.head(5))

# %%

train_image_generator = ImageDataGenerator(rescale=1./255, 
                                            rotation_range=180, 
                                            width_shift_range = [-0.1, 0.1], 
                                            height_shift_range = [-0.1, 0.1], 
                                            shear_range = 0.1, 
                                            zoom_range = 0.25, 
                                            horizontal_flip = True, 
                                            vertical_flip = True) # Generator for our training data
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
earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
wandb_callback = WandbCallback(input_type='image', labels=[0,1], save_weights_only=True)

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy

# %%
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = MODEL_LIST[MODEL](weights="imagenet", include_top=False, pooling='avg')

    if FREEZE:
        base_model.trainable = False
    else:
        base_model.trainable = True
    out = layers.Dropout(0.5)(base_model.output)
    out = layers.Dense(1, activation="sigmoid")(out)
    model = models.Model(base_model.input, out)
    model.compile(loss = focal_loss(), optimizer=optimizers.Adam(learning_rate=0.0001), metrics=[tf.keras.metrics.AUC()])
print("Build Network !")
# %%
model.fit(train_generator, \
            epochs=EPOCHS, \
            validation_data=val_generator, \
            callbacks = [reducelr, earlystop, wandb_callback], \
            workers=24,
            max_queue_size=32)

# %%
# %%
result = model.predict(test_generator, verbose=1)
# %%
SP_CSV_PATH = os.path.join(ROOT, "sample_submission.csv")
sample_csv = pd.read_csv(SP_CSV_PATH)
sample_csv['target'] = result
sample_csv.to_csv(f"./{date}_submission_{MODEL}_{FREEZE}.csv", index=False)

