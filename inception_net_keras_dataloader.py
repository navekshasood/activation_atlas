import cv2
import glob
import numpy as np
import torch

import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize

from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_PATH = "/N/slate/soodn/ILSVRC/Data/CLS-LOC/train/"
classes = glob.glob(BASE_PATH+"*")

tf.compat.v1.keras.backend.set_image_data_format('channels_last')
base_model = InceptionV3(weights='imagenet', include_top=False)
base_model.trainable = True

image_size = (256, 256)
batch_size = 64
n_classes = 1000

train_ds = tf.keras.utils.image_dataset_from_directory(
  BASE_PATH,
  labels = "inferred",
  label_mode = 'categorical',
  validation_split=0.3,
  subset="training",
  seed=123,
  image_size=image_size,
  batch_size=batch_size)

train_ds_subset = train_ds.take(500)
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   BASE_PATH,
#   validation_split=0.3,
#   subset="validation",
#   seed=123,
#   image_size=image_size,
#   batch_size=batch_size)
print ("Dataset made.")

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print ("Model compiled.")

epochs = 10
model.fit(train_ds_subset, epochs=epochs, batch_size=batch_size)
model.save('trained_model_epochs100')
print("Model trained.")

