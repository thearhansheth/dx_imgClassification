#importing required packages onto virtual env
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import cv2 as cv

#Loading datasets from local directory
train_ds = keras.utils.image_dataset_from_directory(
    directory='data3a/training',
    )

validation_ds = keras.utils.image_dataset_from_directory(
    directory='data3a/validation',
    )


#using variables to store batch data for easier access
for image_batch, labels_batch in train_ds:
  train_image = image_batch
  #print("Image Batch Shape:", image_batch.shape)
  train_label = labels_batch
  #print("Labels Batch Shape:", labels_batch.shape)
  break


for image_batch, labels_batch in validation_ds:
  test_image = image_batch
  #print(image_batch.shape)
  test_label = labels_batch
  #print(labels_batch.shape)
  break

#Converting into 2d arrays for scikit-learn
nsamples, nx, ny, nrgb = train_image.shape
train_image2 = train_image.reshape((nsamples, nx * ny * nrgb))

nsamples, nx, ny, nrgb = test_image.shape
test_image2 = test_image.reshape((nsamples, nx * ny * nrgb))

