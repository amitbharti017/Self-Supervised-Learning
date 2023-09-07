#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

np.random.seed(2110)

### Hyperparameters for training
#### Patch creation
BUFFER_SIZE = 1024
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (64,64,3)

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.005
EPOCHS = 350

#### Image Patch Dimension

IMAGE_SIZE = 64
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_SIZE//PATCH_SIZE)**2
MASK_PROPORTION = 0.75

#### Encoder and Decoder
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 512
DEC_PROJECTION_DIM = 256
ENC_NUM_HEADS = 6
ENC_LAYERS = 6
DEC_NUM_HEADS = 6
DEC_LAYERS = 3
ENC_TRANSFORMER_UNITS = [ENC_PROJECTION_DIM*2,ENC_PROJECTION_DIM]
DEC_TRANSFORMER_UNITS = [DEC_PROJECTION_DIM*2,DEC_PROJECTION_DIM]

### Data Loading

train_dir = "../../../../storage/amit/tiny_imagenet/train/"
val_dir = "../../../../storage/amit/tiny_imagenet/validation/"
train_image_path = [train_dir+file for file in os.listdir(train_dir) if file.endswith(".JPEG")]
val_image_path = [val_dir+file for file in os.listdir(val_dir) if file.endswith(".JPEG")]

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image,channels=3)
    return image

train_dataset = tf.data.Dataset.from_tensor_slices(train_image_path)
val_dataset = tf.data.Dataset.from_tensor_slices(val_image_path)

# Map the load functions to the datasets
train_dataset = train_dataset.map(load_image)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

val_dataset = val_dataset.map(load_image)
val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)


#### Augmentation
def train_augmentation():
    train_augmentation_model = tf.keras.Sequential(
    [
        layers.Rescaling(1/255.0),
        layers.Resizing(INPUT_SHAPE[0]+20,INPUT_SHAPE[1]+20),
        layers.RandomCrop(IMAGE_SIZE,IMAGE_SIZE),
        layers.RandomFlip("horizontal")
    ])
    return train_augmentation_model

def test_augmentation():
    test_augmentation_model=tf.keras.Sequential(
    [
        layers.Rescaling(1/255.0)
    ])
    return test_augmentation_model    


### Patch Extraction
class Patches(layers.Layer):
    def __init__(self,patch_size=PATCH_SIZE,**kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.resize = layers.Reshape((-1,patch_size*patch_size*3))
    def __call__(self,images):
        patches = tf.image.extract_patches(
                    images=images,
                    sizes = [1,self.patch_size, self.patch_size,1],
                    strides = [1,self.patch_size,self.patch_size,1],
                    rates = [1,1,1,1],
                    padding = "VALID"
                    )
        patches = self.resize(patches)
        return patches


### Patch Embedding

class PatchEncoder(layers.Layer):
    def __init__(self,patch_size=PATCH_SIZE,projection_dim=ENC_PROJECTION_DIM,
                mask_proportion=MASK_PROPORTION,**kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.mask_token = tf.Variable(tf.random.normal([1,patch_size*patch_size*3],seed=2110))

    def build(self,input_shape):
        (_,self.num_patches,self.patch_area) = input_shape
        self.projection = layers.Dense(units=self.projection_dim,kernel_initializer="glorot_uniform")
        self.position_embedding = layers.Embedding(input_dim=self.num_patches,output_dim = self.projection_dim,embeddings_initializer="uniform")
        self.num_mask = int(self.mask_proportion*self.num_patches)
    def call(self,patches):
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0,limit=self.num_patches,delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis,...])
        pos_embeddings = tf.tile(pos_embeddings,[batch_size,1,1])
        patch_embeddings = (self.projection(patches)+pos_embeddings)
        
        mask_indices,unmask_indices = self.get_random_indices(batch_size)
        unmasked_embeddings = tf.gather(patch_embeddings,unmask_indices,axis=1,batch_dims=1)
        unmasked_positions = tf.gather(pos_embeddings,unmask_indices,axis=1,batch_dims=1)
        masked_positions = tf.gather(pos_embeddings,mask_indices,axis=1,batch_dims=1) 
        mask_tokens = tf.repeat(self.mask_token,repeats=self.num_mask,axis=0)
        mask_tokens = tf.repeat(mask_tokens[tf.newaxis,...],repeats=batch_size,axis=0)
        masked_embeddings = self.projection(mask_tokens)+masked_positions
        return(unmasked_embeddings,
                   masked_embeddings,   
                   unmasked_positions,  
                   mask_indices,        
                   unmask_indices,      
                  )    
    def get_random_indices(self,batch_size):
        rand_indices = tf.argsort(tf.random.uniform(shape = (batch_size,self.num_patches)),axis=-1)
        mask_indices = rand_indices[:,:self.num_mask]
        unmask_indices = rand_indices[:,self.num_mask:]
        return mask_indices, unmask_indices


### Implementation of multilayer perceptron(MLP)

def mlp(x,hidden_units,dropout):
    for units in hidden_units:
        x = layers.Dense(units,activation=tf.nn.gelu,kernel_initializer="glorot_uniform")(x)
        x = layers.Dropout(dropout)(x)
    return x

### MAE Encoder
def encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS):
    inputs = layers.Input((None,ENC_PROJECTION_DIM))
    x = inputs
    for _ in range(num_layers):
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,key_dim=ENC_PROJECTION_DIM,dropout=0.2,kernel_initializer="glorot_uniform")(x1,x1)
        x2 = layers.Add()([attention_output,x])
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
        x3 = mlp(x3,hidden_units=ENC_TRANSFORMER_UNITS,dropout=0.2)
        x = layers.Add()([x3,x2])
    outputs = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    return tf.keras.Model(inputs,outputs)


### MAE Decoder
def decoder(num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS,image_size=IMAGE_SIZE):
    inputs = layers.Input((NUM_PATCHES,ENC_PROJECTION_DIM))
    x = layers.Dense(DEC_PROJECTION_DIM)(inputs)
    for _ in range(num_layers):
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads,key_dim=DEC_PROJECTION_DIM,dropout=0.2,kernel_initializer="glorot_uniform")(x1,x1)
        x2 = layers.Add()([attention_output,x])
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
        x3 = mlp(x3,hidden_units=DEC_TRANSFORMER_UNITS,dropout=0.2)
        x = layers.Add()([x3,x2])
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    x = layers.Flatten()(x)
    pre_final = layers.Dense(units=image_size*image_size*3, activation ="sigmoid",kernel_initializer="glorot_uniform")(x)
    outputs = layers.Reshape((image_size,image_size,3))(pre_final)
    return tf.keras.Model(inputs,outputs)


### MAE Trainer
class MaskedAutoencoder(tf.keras.Model):
    def __init__(self,train_augmentation,test_augmentation,patch_layer,patch_encoder,encoder,decoder,**kwargs):
        super().__init__(**kwargs)
        self.train_augmentation = train_augmentation
        self.test_augmentation = test_augmentation
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
    def calculate_loss(self,images,test=False):
        if test:
            augmented_images = self.test_augmentation(images)
        else:
            augmented_images = self.train_augmentation(images)
        patches = self.patch_layer(augmented_images)
        (unmasked_embeddings,masked_embeddings,unmasked_positions,mask_indices,unmask_indices) = self.patch_encoder(patches)
        encoder_outputs = self.encoder(unmasked_embeddings)
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs,masked_embeddings],axis=1)
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)
        
        loss_patch = tf.gather(patches,mask_indices,axis=1,batch_dims=1)
        loss_output = tf.gather(decoder_patches,mask_indices,axis=1,batch_dims=1)
        total_loss = self.compiled_loss(loss_patch,loss_output)
        return total_loss, loss_patch, loss_output
    def train_step(self,images):
        with tf.GradientTape() as tape:
            total_loss,loss_patch,loss_output = self.calculate_loss(images)
        train_vars = [self.train_augmentation.trainable_variables,
                      self.patch_layer.trainable_variables,
                      self.patch_encoder.trainable_variables,
                      self.encoder.trainable_variables,
                      self.decoder.trainable_variables,]
        grads = tape.gradient(total_loss,train_vars)
        tv_list = []
        for (grad,var) in zip(grads,train_vars):
            for g,v in zip(grad,var):
                tv_list.append((g,v))
        self.optimizer.apply_gradients(tv_list)
        self.compiled_metrics.update_state(loss_patch,loss_output)
        return {m.name:m.result() for m in self.metrics}
    def test_step(self,images):
        total_loss,loss_patch,loss_output=self.calculate_loss(images,test=True)
        self.compiled_metrics.update_state(loss_patch,loss_output)
        return {m.name:m.result() for m in self.metrics}

### Model Initialization
mae_model = MaskedAutoencoder(
                            train_augmentation = train_augmentation(),
                            test_augmentation = test_augmentation(),
                            patch_layer = Patches(),
                            patch_encoder = PatchEncoder(),
                            encoder = encoder(),
                            decoder = decoder(),
                             )


### Learning rate scheduler

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,learning_rate_base,total_steps,warmup_learning_rate,warmup_steps):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)
    def __call__(self,step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
            
        cos_annealed_lr = tf.cos(self.pi*(tf.cast(step,tf.float32)-self.warmup_steps)
                                 /float(self.total_steps - self.warmup_steps))
        learning_rate = 0.5*self.learning_rate_base*(1+cos_annealed_lr)
        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError("Learning_rate_base must be larger or equal to warmup_learning_rate")
            slope = (self.learning_rate_base - self.warmup_learning_rate)/self.warmup_steps
            warmup_rate = slope*tf.cast(step,tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(step<self.warmup_steps,warmup_rate,learning_rate)
        return tf.where(step > self.total_steps,0.0,learning_rate)

total_steps = int((len(train_image_path)/BATCH_SIZE)*EPOCHS)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps*warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
                    learning_rate_base = LEARNING_RATE,
                    total_steps = total_steps,
                    warmup_learning_rate = 0.0,
                    warmup_steps = warmup_steps)
#### Tensorboard logger and early stopping
file_name="tiny_imagenet8"
tensorboard = TensorBoard(log_dir="logs/final_model/{}".format(file_name))

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=15,monitor="val_loss",min_delta=0.0001),
    tensorboard
]
### Model compilation and training
optimizer = tf.keras.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)
mae_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])
history = mae_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset,callbacks=callbacks)
mae_model.save_weights("model_weights/final_model/tiny_imagenet/model8")

