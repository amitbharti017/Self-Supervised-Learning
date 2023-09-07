#!/usr/bin/env python
# coding: utf-8
#### 1. Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.stats import mode
import albumentations as A
from patchify import patchify

#### 2. Dataset Path
jigsaw_images = np.load("../../../../../../storage/amit/dataset/COCO/Ex2_COCO_pascal_voc_jigsaw_images.npy")

model1=load_model('model/jigsaw_COCO1.h5')
model2=load_model('model/jigsaw_COCO2.h5')
model3=load_model('model/jigsaw_COCO3.h5')
model4=load_model('model/jigsaw_COCO4.h5')
model5=load_model('model/jigsaw_COCO5.h5')
def batch_model_prediction(model, jigsaw_imgs):
    y_confi = model.predict(jigsaw_imgs, verbose=0)
    y_preds = np.argmax(y_confi, axis=1)
    return y_confi, y_preds

batch_size = 64
num_images = len(jigsaw_images)
num_batches = (num_images + batch_size - 1) // batch_size

ensemble_predictions = []
ensemble_confidences = []

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_images)

    batch_jigsaw_imgs = jigsaw_images[start_idx:end_idx]

    y_con1, y_pred1 = batch_model_prediction(model1, batch_jigsaw_imgs)
    y_con2, y_pred2 = batch_model_prediction(model2, batch_jigsaw_imgs)
    y_con3, y_pred3 = batch_model_prediction(model3, batch_jigsaw_imgs)
    y_con4, y_pred4 = batch_model_prediction(model4, batch_jigsaw_imgs)
    y_con5, y_pred5 = batch_model_prediction(model5, batch_jigsaw_imgs)
    y_predictions = np.stack((y_pred1, y_pred2, y_pred3, y_pred4, y_pred5), axis=1)
    y_confidences = np.stack((y_con1, y_con2, y_con3, y_con4, y_con5), axis=1)
    if batch_idx == 0:
        ensemble_predictions = y_predictions
        ensemble_confidences = y_confidences
    else:
        ensemble_predictions = np.concatenate((ensemble_predictions, y_predictions), axis=0)
        ensemble_confidences = np.concatenate((ensemble_confidences, y_confidences), axis=0)
ensemble_predictions = np.array(ensemble_predictions)
ensemble_confidences = np.array(ensemble_confidences)
mean_confidences = np.mean(ensemble_confidences, axis=1)
mode_predictions,_ = mode(ensemble_predictions,axis=1)
mode_predictions = mode_predictions[:,0]
np.save('Ex2_COCO_pascal_voc_confidences_mean', mean_confidences)
np.save('Ex2_COCO_pascal_voc_prediction_mode', mode_predictions)

