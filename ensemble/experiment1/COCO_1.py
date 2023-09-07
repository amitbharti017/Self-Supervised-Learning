#!/usr/bin/env python
# coding: utf-8
#### 1. Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm
import albumentations as A
import tensorflow as tf
from tensorflow.keras.models import load_model
#### 2. Loading Dataset
noisy_images = np.load('../../../../../../storage/amit/dataset/COCO/Ex1_COCO_cityscapes_noisy_images.npy')
model1 = load_model('model/COCO1.h5')
model2 = load_model('model/COCO2.h5')
model3 = load_model('model/COCO3.h5')
model4 = load_model('model/COCO4.h5')
model5 = load_model('model/COCO5.h5')
## Ensemble
import numpy as np
batch_size = 64
num_images = len(noisy_images)
num_batches = (num_images + batch_size - 1) // batch_size
ensemble_predictions = []
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_images)

    batch_images = noisy_images[start_idx:end_idx]
    batch_images = np.stack(batch_images)  # Convert the batch to a numpy array

    y_pred1 = model1.predict(batch_images, verbose=0)
    y_pred2 = model2.predict(batch_images, verbose=0)
    y_pred3 = model3.predict(batch_images, verbose=0)
    y_pred4 = model4.predict(batch_images, verbose=0)
    y_pred5 = model5.predict(batch_images, verbose=0)

    y_predictions = np.stack((y_pred1, y_pred2, y_pred3, y_pred4, y_pred5), axis=1)

    if batch_idx == 0:
        ensemble_predictions = y_predictions
    else:
        ensemble_predictions = np.concatenate((ensemble_predictions, y_predictions), axis=0)

confidences = np.array(ensemble_predictions)
confidences_mean = np.mean(confidences, axis=1)
confidences_std = np.std(confidences, axis=1)
np.save('Ex1_COCO_cityscapes_confidences_mean', confidences_mean)
np.save('Ex1_COCO_cityscapes_confidences_std', confidences_std)

