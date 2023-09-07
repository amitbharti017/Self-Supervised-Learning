#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import random
np.random.seed(2110)

import tensorflow as tf
import cv2
import keras
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Dropout,BatchNormalization,Input,GaussianNoise
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau

#### Loading Dataset

(x_train, _), (x_test, _) = keras.datasets.cifar100.load_data()

def add_noise(images):
    '''
    Given an array of images in a shape (None, Height, Width, Channels),
    this function adds gaussian noise to every channel of the image
    '''
    #adding noise to images
    noise = np.random.normal(0,0.2,size=images.shape)
    noisy_img = images+noise 
    noisy_img = np.clip(noisy_img, 0., 1.)
    return noisy_img

x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.
x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

#### 6. Model Building
TRAIN_SIZE = 32
start = Input(shape=(TRAIN_SIZE,TRAIN_SIZE,3),dtype=tf.float32)
#to make the model accept variable size inputs, we set the input shape parameter to (None,None,3)
#### Constructing U-Net Architecture
 #Encoder
c1 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer = "he_normal",padding="same")(start)
c1 = BatchNormalization()(c1)
c1 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c1)
c1 = BatchNormalization()(c1)
p1=tf.keras.layers.MaxPooling2D((2,2))(c1)
p1 = tf.keras.layers.Dropout(0.2)(p1)

c2=tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p1)
c2 = BatchNormalization()(c2)
c2=tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c2)
c2 = BatchNormalization()(c2)
p2= tf.keras.layers.MaxPooling2D((2,2))(c2)
p2 = tf.keras.layers.Dropout(0.3)(p2)

c3=tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p2)
c3 = BatchNormalization()(c3)
c3=tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c3)
c3 = BatchNormalization()(c3)

u4 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding="same")(c3)
u4=tf.keras.layers.concatenate([u4,c2])
u4=tf.keras.layers.Dropout(0.3)(u4)
c4=tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u4)
c4 = BatchNormalization()(c4)
c4=tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c4)
c4 = BatchNormalization()(c4)

u5=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding="same")(c4)
u5=tf.keras.layers.concatenate([u5,c1])
u5=tf.keras.layers.Dropout(0.2)(u5)
c5=tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u5)
c5 = BatchNormalization()(c5)
c5=tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c5)
c5 = BatchNormalization()(c5)

outputs=tf.keras.layers.Conv2D(3,(1,1),activation="sigmoid")(c5)

model_cifar100=tf.keras.Model(inputs=[start],outputs=[outputs])
model_cifar100.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()])
model_cifar10.summary()

#### Callback functions

checkpointer= tf.keras.callbacks.ModelCheckpoint("./models/ensemble/cifar100_5.h5",verbose=1,save_best_only=True)

file_name="denoise_cifar100_5"
tensorboard = TensorBoard(log_dir="logs/ensemble/{}".format(file_name))

callbacks=[
    EarlyStopping(patience=12,monitor="val_loss",min_delta=0.0001),
    ReduceLROnPlateau(monitor="val_loss",factor=0.8,patience=5,verbose=1),
    tensorboard,
    checkpointer
]

#### Train Model
history = model_cifar100.fit(x_train_noisy,x_train,validation_data=(x_test_noisy,x_test),batch_size=32,epochs=100,callbacks=callbacks)

