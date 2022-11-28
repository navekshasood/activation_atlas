import cv2
import glob
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize

from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from umap_utils import reduce_dim_umap, normalize_layout 
from render_layout_utils import render_layout
from objective_utils import get_feature_extractor
from image_utils import deprocess_image

import warnings
warnings.filterwarnings("ignore")

BASE_PATH = "/N/slate/soodn/ILSVRC/Data/CLS-LOC/train/"
classes = glob.glob(BASE_PATH+"*")

tf.compat.v1.keras.backend.set_image_data_format('channels_last')
base_model = InceptionV3(weights='imagenet', include_top=False)
base_model.trainable = True

image_size = (256, 256, 3)
n_classes = 20
images_per_class = 100

def create_train_data(classes):
    trainX, trainY = [], []
    for i, class_path in enumerate(classes[0:n_classes]):
        print (f"Class {i+1}")
        for image_path in (glob.glob(class_path+"/*"))[:images_per_class]:
            image = cv2.imread(image_path)
            resized_image = resize(image, image_size)
            trainX.append(resized_image)
            trainY.append(i)

    # label_encoder = LabelEncoder()
    # integer_encoded = label_encoder.fit_transform(trainY)
    trainY = np.array(trainY)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = trainY.reshape(len(trainY), 1)
    trainY_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return np.array(trainX), trainY_onehot_encoded

trainX, trainY = create_train_data(classes)
print (trainX.shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 16
epochs = 5

# model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
# model.save('trained_model_epochs100')

layer = "mixed10"
model = keras.models.load_model("trained_model_epochs100")

layer_output = get_feature_extractor(model, layer)

activations = layer_output(trainX)
print (activations.shape)

# for X in trainX:
#     X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
#     activations.append(layer_output(X))

activations = np.array(activations)
flattened_activations = activations.reshape(activations.shape[0], activations.shape[1]*activations.shape[2]*activations.shape[3])
activations_reduced = reduce_dim_umap(flattened_activations)
activations_normalized = normalize_layout(activations_reduced)

# Feature visualization
xs = activations_normalized[:, 0]
ys = activations_normalized[:, 1]
plt.scatter(xs, ys, label = "trainY")
plt.savefig('umap.png')

def whiten(full_activations):
    correl = np.matmul(full_activations.T, full_activations) / len(full_activations)
    correl = correl.astype("float32")
    S = np.linalg.inv(correl)
    S = S.astype("float32")
    return S

S = whiten(raw_activations)

canvas = render_layout(model, layer, activations, xs, ys, S, raw_activations, n_steps=512, grid_size=(8, 8))
print (canvas.shape, np.max(canvas), np.min(canvas))
from PIL import Image

im = Image.fromarray(deprocess_image(canvas))
im.save("canvas.png")