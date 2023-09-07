#!/usr/bin/env python
# coding: utf-8

#### 1. Importing Libraries
import numpy as np
import os
import random
import cv2
from tqdm import tqdm
from scipy.stats import mode
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import albumentations as A
from patchify import patchify
#### 2. Loading all the files

jigsaw_images = np.load("../../../../../../storage/amit/dataset/COCO/Ex2_COCO_cityscapes_jigsaw_images.npy")

model=load_model('model/jigsaw_COCO1.h5')

def predict_mc_dropout(model, jigsaw_images, num_mc_samples):
    batch_size = 64
    num_images = len(jigsaw_images)
    num_batches = (num_images + batch_size - 1) // batch_size
    mc_predictions = []
    mc_confidences = []
    for i in tqdm(range(num_mc_samples)):
        y_predictions = []
        y_confidences = []
        for b in range(num_batches):
            batch_start = b * batch_size
            batch_end = min((b + 1) * batch_size, num_images)
            batch_size_actual = batch_end - batch_start
            jigsaw_images_batch = jigsaw_images[batch_start:batch_end]
            model_output = model(jigsaw_images_batch, training=True)
            y_confi = model_output.numpy() 
            y_pred = np.argmax(y_confi, axis=1)
            y_predictions.extend(y_pred.tolist())
            y_confidences.extend(y_confi.tolist())
        mc_predictions.append(y_predictions)
        mc_confidences.append(y_confidences)
    mc_predictions = np.array(mc_predictions)
    mc_confidences = np.array(mc_confidences)
    return mc_predictions, mc_confidences

# Perform Monte Carlo dropout sampling
num_mc_samples = 10
mc_predictions,mc_confidences = predict_mc_dropout(model, jigsaw_images, num_mc_samples)
mean_confidences = np.mean(mc_confidences, axis=0)
mode_predictions,_ = mode(mc_predictions)
mode_predictions = mode_predictions[0,:]
np.save('Ex2_COCO_cityscapes_confidences_mean',mean_confidences)
np.save('Ex2_COCO_cityscapes_prediction_mode', mode_predictions)

