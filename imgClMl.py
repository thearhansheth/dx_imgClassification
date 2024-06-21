#importing required packages onto virtual env
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2 as cv

#loading dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#Data pre-processing
#Normalization of images. Making sure all images are reshaped into uniform range
x_train = x_train/255.0
x_test = x_test/255.0

#Converting into 2d arrays since scikit learn except 2d array model for .fit()
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))

nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

model = RandomForestClassifier()
model.fit(x_train2, y_train.ravel())
y_pred = model.predict(x_test2)

print("Accuracy Score: ", accuracy_score(y_pred, y_test))
print("Classification Report: ", classification_report(y_pred, y_test))
print("Confusion Matrix: ", confusion_matrix(y_pred, y_test))