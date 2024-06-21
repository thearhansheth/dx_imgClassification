#importing required packages onto virtual env
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import cv2 as cv

#loading dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print("Preprocessing:")
print(x_train.shape)
print(x_test.shape)

#Data pre-processing
#Normalization of images. Making sure all images are reshaped into uniform range
x_train = x_train/255.0
x_test = x_test/255.0

#Converting into 2d arrays since scikit learn except 2d array model for .fit()
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))

nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

print("Post Processing:")
print(x_train2.shape)
print(x_test2.shape)

