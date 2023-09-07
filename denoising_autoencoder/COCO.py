#!/usr/bin/env python
# coding: utf-8
#### 1. Importing Libraries

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

import datetime
import os
from tensorboard.plugins import projector

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')
#### 2. Loading Dataset
train_dir = "../../datasets/COCO/train2017/"
test_dir = "../../datasets/COCO/validation2017/"
train_image_path = [train_dir+file for file in os.listdir(train_dir) if file.endswith(".jpg")]
test_image_path = [test_dir+file for file in os.listdir(test_dir) if file.endswith(".jpg")]
#### 5. Data Generator

class DataGenerator(keras.utils.Sequence):
    "Generates data for keras"
    def __init__(self,path,batch_size=32,n_channels=3,shuffle=True):
        "Initialization"
        self.batch_size=batch_size
        self.path =path
        self.n_channels=n_channels
        self.shuffle =shuffle
        self.on_epoch_end()
        
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.path)/self.batch_size))
    
    def __getitem__(self,index):
        "Generate one batch of data"
        #Generate indexes of the batch
        indexes= self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        current_batch = [self.path[k] for k in indexes]
        X = self.__data_generation(current_batch)
        return X
    
    def on_epoch_end(self):
        "Update indexes after each epoch"
        self.indexes=np.arange(len(self.path))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self,current_batch):
        "Generates data containing batch_size samples"        
        #Generate data
        #array_storage_and_normalization
        batch_images=[]
        for file in current_batch:
            image = plt.imread(file)
            try:  
                if len(image[0][0]) == self.n_channels:
                    pass
            except:
                image = np.stack((image,)*self.n_channels, axis=-1)
            image_reduced = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            image_reduced = np.array(image_reduced).astype(np.float32)
            image_reduced=image_reduced/255.0 #Normalizing the image pixel values to be between 0 and 1
            batch_images.append(image_reduced)
        batch_images = np.array(batch_images)
        '''
        Given an array of images in a shape (None, Height, Width, Channels),
        this part adds gaussian noise to every channel of the image
        '''
         #creating a matrix for inducing noise in all the channels
      
        noise = np.random.normal(0,0.2,size=batch_images.shape)
        #adding noise to images
        noisy_img = batch_images+noise
        noisy_img = np.clip(noisy_img, 0., 1.)            
        return noisy_img,batch_images


#### 6. Model Building
TRAIN_SIZE = 256
start = Input(shape=(TRAIN_SIZE,TRAIN_SIZE,3),dtype=tf.float32)
#to make the model accept variable size inputs, we set the input shape parameter to (None,None,3)
##### Constructing U-Net Architecture
#Encoder
c1 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer = "he_normal",padding="same")(start)
c1 = BatchNormalization()(c1)
c1 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c1)
c1 = BatchNormalization()(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
p1 = tf.keras.layers.Dropout(0.1)(p1)

c2 = tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p1)
c2 = BatchNormalization()(c2)
c2 = tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c2)
c2 = BatchNormalization()(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
p2 = tf.keras.layers.Dropout(0.2)(p2)

c3 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p2)
c3 = BatchNormalization()(c3)
c3 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c3)
c3 = BatchNormalization()(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
p3 = tf.keras.layers.Dropout(0.3)(p3)

c4 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p3)
c4 = BatchNormalization()(c4)
c4 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c4)
c4 = BatchNormalization()(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
p4 = tf.keras.layers.Dropout(0.3)(p4)

c5 = tf.keras.layers.Conv2D(512,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(p4)
c5 = BatchNormalization()(c5)
c5 = tf.keras.layers.Conv2D(512,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c5)
c5 = BatchNormalization()(c5)

#expansive path
u6 = tf.keras.layers.Conv2DTranspose(256,(2,2),strides=(2,2),padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
u6 = tf.keras.layers.Dropout(0.3)(u6)
c6 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u6)
c6 = BatchNormalization()(c6)
c6 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c6)
c6 = BatchNormalization()(c6)

u7 = tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
u7 = tf.keras.layers.Dropout(0.3)(u7)
c7 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u7)
c7 = BatchNormalization()(c7)
c7 = tf.keras.layers.Conv2D(128,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c7)
c7 = BatchNormalization()(c7)

u8 = tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
u8 = tf.keras.layers.Dropout(0.2)(u8)
c8 = tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u8)
c8 = BatchNormalization()(c8)
c8 = tf.keras.layers.Conv2D(64,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c8)
c8 = BatchNormalization()(c8)

u9 = tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
u9 = tf.keras.layers.Dropout(0.1)(u9)
c9 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(u9)
c9 = BatchNormalization()(c9)
c9 = tf.keras.layers.Conv2D(32,(3,3),activation="relu",kernel_initializer="he_normal",padding="same")(c9)
c9 = BatchNormalization()(c9)

outputs = tf.keras.layers.Conv2D(3,(1,1),activation="sigmoid")(c9)

model_COCO32=tf.keras.Model(inputs=[start],outputs=[outputs])
model_COCO32.compile(optimizer=Adam(learning_rate=0.0001),loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()])
model_COCO32.summary()
checkpointer= tf.keras.callbacks.ModelCheckpoint("./models/ensemble/COCO5.h5",verbose=1,save_best_only=True)

file_name="denoise_COCO5"
tensorboard = TensorBoard(log_dir="logs/ensemble/{}".format(file_name))

callbacks=[
    EarlyStopping(patience=12,monitor="val_loss",min_delta=0.0001),
    ReduceLROnPlateau(monitor="val_loss",factor=0.8,patience=5,verbose=1),
    tensorboard,
    checkpointer
]

params = {'batch_size': 32,
          'n_channels': 3,
          'shuffle': True}

#### Train Model
train_generator = DataGenerator(train_image_path,**params)
val_generator = DataGenerator(test_image_path,**params)

history = model_COCO32.fit(train_generator,validation_data=val_generator,epochs=100,callbacks=callbacks)

