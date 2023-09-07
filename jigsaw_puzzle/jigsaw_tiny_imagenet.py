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
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization,Input
from tensorflow.keras.layers import Dense,TimeDistributed,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard,ReduceLROnPlateau,ModelCheckpoint
import albumentations as A
from patchify import patchify
import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

#### 2. Dataset Path
train_dir = "../../../../storage/amit/tiny_imagenet/train/"
val_dir =  "../../../../storage/amit/tiny_imagenet/validation/"
train_image_path = [train_dir+file for file in os.listdir(train_dir) if file.endswith(".JPEG")]
val_image_path = [val_dir+file for file in os.listdir(val_dir) if file.endswith(".JPEG")]
#### 3. Loading Permutation File
selected_permutation = np.load("permutation.npy")
#### 4. Image and patch dimention to be used
ORIGINAL_IMAGE_SIZE = 256
CROPPED_IMAGE_SIZE = 225
# Define the dimensions of the jigsaw puzzle
PATCH_ORIGINAL_SIZE=75
PUZZLE_PATCH = 64
PUZZLE_SIZE = 3
PUZZLE_DIM = PUZZLE_PATCH*PUZZLE_SIZE
# Define the number of patches and the patch dimensions
NUM_PATCHES = PUZZLE_SIZE * PUZZLE_SIZE
PATCH_DIM = PUZZLE_DIM // PUZZLE_SIZE
CHANNELS = 3
#### 5. Loading Image and Augmentation applied to loaded image
aug_original_image = A.Compose([A.RandomCrop(always_apply=True, p=1.0, height=CROPPED_IMAGE_SIZE, width=CROPPED_IMAGE_SIZE)])
aug_tile = A.Compose([A.RandomCrop( p=1.0, height=PUZZLE_PATCH, width=PUZZLE_PATCH),
                     A.RGBShift(p=1, r_shift_limit=(-0.2, 0.2), g_shift_limit=(-0.2, 0.2), b_shift_limit=(-0.2, 0.2)),
                     A.ToGray(p=0.3)
                     ])


#### 8.Datagenerator for model
class DataGenerator(keras.utils.Sequence):
    def __init__(self,path,permutation,original_image_size,cropped_image_size,patch_original_size,
                 puzzle_patch,puzzle_size,puzzle_dim,num_patches,patch_dim,batch_size=4,n_channels=3,shuffle=True):
        self.path =path
        self.permutation = permutation
        self.original_image_size = original_image_size
        self.cropped_image_size = cropped_image_size
        self.patch_original_size = patch_original_size
        self.puzzle_patch = puzzle_patch
        self.puzzle_size = puzzle_size
        self.puzzle_dim = puzzle_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.batch_size=batch_size
        self.n_channels=n_channels
        self.shuffle =shuffle
        self.on_epoch_end()
        
    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.ceil(len(self.path)/self.batch_size))
    
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
    
    def resize_crop(self, img):
        height,width,_=img.shape
        if width<=height:
            ratio = height/float(width)
            changed_height = int(np.ceil(ratio*self.original_image_size))
            image_reduced = cv2.resize(img, dsize=(self.original_image_size, changed_height), interpolation=cv2.INTER_CUBIC)
        else:
            ratio = width/float(height)
            changed_width = int(np.ceil(ratio*self.original_image_size))
            image_reduced = cv2.resize(img, dsize=(changed_width, self.original_image_size), interpolation=cv2.INTER_CUBIC)
        augmented = aug_original_image(image = image_reduced)
        return augmented["image"]
    
    def load_image(self,image_path):
        image = plt.imread(image_path)
        image_enhanced = cv2.resize(image, dsize=(self.original_image_size, self.original_image_size), interpolation=cv2.INTER_CUBIC)
        try:  
            if len(image_enhanced[0][0]) == self.n_channels:
                pass
        except:
            image_enhanced = np.stack((image_enhanced,)*self.n_channels, axis=-1)
        image_reduced = self.resize_crop(image_enhanced)
        image = np.array(image_reduced).astype(np.float32)
        image=image/255.
        return image 
    
    def jigsaw(self,image):
        label = np.random.randint(len(self.permutation))  
        tiles = [None] * self.num_patches
        patch_tiles=patchify(image,(self.patch_original_size,self.patch_original_size,self.n_channels),step=self.patch_original_size)
        for i in range(patch_tiles.shape[0]):
            for j in range(patch_tiles.shape[1]):
                single_patch_img = patch_tiles[i, j, 0, :, :, :]
                augmented = aug_tile(image = single_patch_img)
                augmented_tile = augmented["image"]
                tiles[3*i+j] = augmented_tile      
        jigsaw_image = [tiles[selected_permutation[label][t]] for t in range(9)]
        jigsaw_image= np.asarray(jigsaw_image)
        return jigsaw_image,label
            
    def __data_generation(self,current_batch):
        "Generates data containing batch_size samples"        
        batch_images=[]
        batch_labels=[]
        for file in current_batch:
            image = self.load_image(file)
            jigsaw_image,label=self.jigsaw(image)
            batch_images.append(jigsaw_image)
            batch_labels.append(label)
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        return batch_images,batch_labels


### Alex-Net Architecture

def patch_features(shape=(PUZZLE_PATCH,PUZZLE_PATCH,CHANNELS)):
    patch_model = tf.keras.models.Sequential()
    patch_model.add(tf.keras.Input(shape=shape))
    patch_model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation="relu", kernel_initializer="he_normal", padding="same"))
    patch_model.add(BatchNormalization())
    patch_model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    patch_model.add(Dropout(0.1))
    patch_model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation="relu",kernel_initializer="he_normal",padding="same"))
    patch_model.add(BatchNormalization())
    patch_model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    patch_model.add(Dropout(0.2))
    patch_model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu",kernel_initializer="he_normal",padding="same"))
    patch_model.add(BatchNormalization())
    patch_model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu",kernel_initializer="he_normal",padding="same"))
    patch_model.add(BatchNormalization())
    patch_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation="relu",kernel_initializer="he_normal",padding="same"))
    patch_model.add(BatchNormalization())
    patch_model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    patch_model.add(Dropout(0.3))
    patch_model.add(Flatten())
    patch_model.add(Dense(4608,activation="relu"))
    return patch_model

def jigsaw(shape=(NUM_PATCHES,PUZZLE_PATCH,PUZZLE_PATCH,CHANNELS)):
    patch_feature_extraction = patch_features(shape[1:])
    jigsaw_model = tf.keras.models.Sequential()
    jigsaw_model.add(TimeDistributed(patch_feature_extraction,input_shape=shape))
    jigsaw_model.add(Flatten())
    jigsaw_model.add(Dense(4608,activation="relu"))
    jigsaw_model.add(Dropout(0.3))
    jigsaw_model.add(Dense(4096,activation="relu"))
    jigsaw_model.add(Dropout(0.3))
    jigsaw_model.add(Dense(1000,activation="softmax"))
    return jigsaw_model

model_tiny_imagenet = jigsaw(shape=(NUM_PATCHES,PUZZLE_PATCH,PUZZLE_PATCH,CHANNELS))
### Training Model
#### Tensorboard logger and early stopping

checkpointer= ModelCheckpoint("./models/ensemble/tiny_imagenet6.h5",verbose=1,save_best_only=True)

file_name="jigsaw_tiny_imagenet6"
tensorboard = TensorBoard(log_dir="logs/ensemble/{}".format(file_name))

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=15,monitor="val_loss",min_delta=0.0001),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.8,patience=5,verbose=1),
    tensorboard,
    checkpointer
]


#### Loss function, metric and optimizer

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

#### Data Generation

params = {'original_image_size' : ORIGINAL_IMAGE_SIZE,
          'cropped_image_size' : CROPPED_IMAGE_SIZE,
          'patch_original_size' : PATCH_ORIGINAL_SIZE,
          'puzzle_patch' : PUZZLE_PATCH,
          'puzzle_size' : PUZZLE_SIZE,
          'puzzle_dim' : PUZZLE_DIM,
          'num_patches' : NUM_PATCHES,
          'patch_dim' : PATCH_DIM,
          'batch_size': 32,
          'n_channels': CHANNELS,
          'shuffle': True}

train_generator = DataGenerator(train_image_path,selected_permutation,**params)
val_generator = DataGenerator(val_image_path,selected_permutation,**params)

model_tiny_imagenet.compile(optimizer=optimizer,loss=loss,metrics=[train_acc_metric])
#### Train Model
history = model_tiny_imagenet.fit(train_generator,validation_data=val_generator,epochs=350,callbacks=callbacks)
