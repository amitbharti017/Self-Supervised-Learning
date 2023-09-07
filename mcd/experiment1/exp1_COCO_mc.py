#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from tqdm import tqdm
import albumentations as A

import tensorflow as tf
from tensorflow.keras.models import load_model

#### Loading Dataset
test_dir_COCO = "../../../../../../storage/amit/COCO/validation2017/"
test_image_path_COCO = [test_dir_COCO+file for file in os.listdir(test_dir_COCO) if file.endswith(".jpg")]

test_dir_city = "../../../../../../storage/amit/test_dataset/cityscapes/"
test_image_path_city = [test_dir_city+file for file in os.listdir(test_dir_city) if file.endswith(".png")]

test_dir_kitti = "../../../../../../storage/amit/test_dataset/KITTI/"
test_image_path_kitti = [test_dir_kitti+file for file in os.listdir(test_dir_kitti) if file.endswith(".png")]

test_dir_pascal = "../../../../../../storage/amit/test_dataset/PASCAL_VOC2017/"
test_image_path_pascal = [test_dir_pascal+file for file in os.listdir(test_dir_pascal) if file.endswith(".jpg")]

#### Image Preprocessing

CROPPED_IMAGE_SIZE = 256
aug_original_image = A.Compose([A.RandomCrop(always_apply=True, p=1.0, height=CROPPED_IMAGE_SIZE, width=CROPPED_IMAGE_SIZE)])

def array_storage_and_normalization_jpg(path):
    image = plt.imread(path)
    try:  
        if len(image[0][0]) == 3:
            pass
    except:
        image = np.stack((image,)*3, axis=-1)
    image = cv2.resize(image, dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
    augmented_image = aug_original_image(image = image)
    augmented_image = np.array(augmented_image["image"]).astype(np.float32)
    augmented_image = augmented_image/255.
    augmented_image = np.clip(augmented_image, 0., 1.)
    return np.array(augmented_image)

def array_storage_and_normalization_png(path):
    image = plt.imread(path)
    try:  
        if len(image[0][0]) == 3:
            pass
    except:
        image = np.stack((image,)*3, axis=-1)
    image = cv2.resize(image, dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
    augmented_image = aug_original_image(image = image)
    augmented_image = np.array(augmented_image["image"]).astype(np.float32)
    augmented_image = np.clip(augmented_image, 0., 1.)
    return np.array(augmented_image)

def add_noise(images):
    noise = np.random.normal(0,0.2,size=images.shape)
    #adding noise to images
    noisy_img = images+noise 
    noisy_img = np.clip(noisy_img, 0., 1.)
    return noisy_img

images_COCO = []
noisy_images_COCO = []
for i in tqdm(range(len(test_image_path_COCO))):
    img = array_storage_and_normalization_jpg(test_image_path_COCO[i])
    noisy_img = add_noise(img)
    images_COCO.append(img)
    noisy_images_COCO.append(noisy_img)
images_COCO = np.array(images_COCO)
noisy_images_COCO = np.array(noisy_images_COCO)

images_city = []
noisy_images_city = []
for i in tqdm(range(len(test_image_path_city))):
    img = array_storage_and_normalization_png(test_image_path_city[i])
    noisy_img = add_noise(img)
    images_city.append(img)
    noisy_images_city.append(noisy_img)
images_city = np.array(images_city)
noisy_images_city = np.array(noisy_images_city)

images_kitti = []
noisy_images_kitti = []
for i in tqdm(range(len(test_image_path_kitti))):
    img = array_storage_and_normalization_png(test_image_path_kitti[i])
    noisy_img = add_noise(img)
    images_kitti.append(img)
    noisy_images_kitti.append(noisy_img)
images_kitti = np.array(images_kitti)
noisy_images_kitti = np.array(noisy_images_kitti)

images_pascal = []
noisy_images_pascal = []
for i in tqdm(range(len(test_image_path_pascal))):
    img = array_storage_and_normalization_jpg(test_image_path_pascal[i])
    noisy_img = add_noise(img)
    images_pascal.append(img)
    noisy_images_pascal.append(noisy_img)
images_pascal = np.array(images_pascal)
noisy_images_pascal = np.array(noisy_images_pascal)

#### Loading Model
model = load_model('model/COCO1.h5')

#### MC Dropout
def perform_multiple_predictions(model, x, num_mc_samples=10):
    tf.keras.backend.set_learning_phase(1)
    y_predictions = []
    for i in range(num_mc_samples):
        y_pred = model(x, training=True)
        y_predictions.append(y_pred)
    tf.keras.backend.set_learning_phase(0)
    return tf.stack(y_predictions)

def predict_mc_dropout(model, noisy_images, batch_size=64, num_mc_samples=10):
    l = len(noisy_images)
    output = np.zeros((num_mc_samples, l, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE, 3))
    for ndx in tqdm(range(0, l, batch_size)):
        x = noisy_images[ndx:min(ndx + batch_size, l)]
        y_predictions = perform_multiple_predictions(model, x, num_mc_samples=num_mc_samples)
        output[:, ndx:ndx + len(x), :, :, :] = y_predictions

    return output
# Perform Monte Carlo dropout sampling
num_mc_samples = 10

confidences_COCO = predict_mc_dropout(model, noisy_images_COCO, num_mc_samples)
confidences_mean_COCO = np.mean(confidences_COCO, axis=0)
confidences_std_COCO = np.std(confidences_COCO, axis=0)
np.save('Ex1_COCO_confidences_mean_4', confidences_mean_COCO)
np.save('Ex1_COCO_confidences_std_4', confidences_std_COCO)
np.save('Ex1_COCO_image_4', images_COCO)
np.save('Ex1_COCO_noisy_images_4',noisy_images_COCO)

confidences_city = predict_mc_dropout(model, noisy_images_city, num_mc_samples)
confidences_mean_city = np.mean(confidences_city, axis=0)
confidences_std_city = np.std(confidences_city, axis=0)
np.save('Ex1_COCO_cityscapes_confidences_mean', confidences_mean_city)
np.save('Ex1_COCO_cityscapes_confidences_std', confidences_std_city)
np.save('Ex1_COCO_cityscapes_image', images_city)
np.save('Ex1_COCO_cityscapes_noisy_images',noisy_images_city)

confidences_kitti = predict_mc_dropout(model, noisy_images_kitti, num_mc_samples)
confidences_mean_kitti = np.mean(confidences_kitti, axis=0)
confidences_std_kitti = np.std(confidences_kitti, axis=0)
np.save('Ex1_COCO_KITTI_confidences_mean_4', confidences_mean_kitti)
np.save('Ex1_COCO_KITTI_confidences_std_4', confidences_std_kitti)
np.save('Ex1_COCO_KITTI_image_4', images_kitti)
np.save('Ex1_COCO_KITTI_noisy_images_4',noisy_images_kitti)

confidences_pascal = predict_mc_dropout(model, noisy_images_pascal, num_mc_samples)
confidences_mean_pascal = np.mean(confidences_pascal, axis=0)
confidences_std_pascal = np.std(confidences_pascal, axis=0)
np.save('Ex1_COCO_pascal_voc_confidences_mean_4', confidences_mean_pascal)
np.save('Ex1_COCO_pascal_voc_confidences_std_4', confidences_std_pascal)
np.save('Ex1_COCO_pascal_voc_image_4', images_pascal)
np.save('Ex1_COCO_pascal_voc_noisy_images_4',noisy_images_pascal)