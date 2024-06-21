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

#loading dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#Data pre-processing
#Normalization of images. Making sure all images are reshaped into uniform range
x_train = x_train/255.0
x_test = x_test/255.0
y_train = y_train.ravel()

#Converting into 2d arrays since scikit learn except 2d array model for .fit()
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))

nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

## Random Forest
model = RandomForestClassifier()
model.fit(x_train2, y_train)
y_pred = model.predict(x_test2)

print("Accuracy Score Random Forest: ", accuracy_score(y_pred, y_test))
#print("Classification Report: ", classification_report(y_pred, y_test))
#print("Confusion Matrix: ", confusion_matrix(y_pred, y_test))


## KNN
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(x_train2, y_train)
y_pred2 = knn.predict(x_test2)
print("Accuracy Score KNN: ", accuracy_score(y_pred2, y_test))


## Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train2, y_train)
y_pred3 = dtc.predict(x_test2)
print("Accuracy Score Decision Tree: ", accuracy_score(y_pred3, y_test))


## Naive Bayes
nb = GaussianNB()
nb.fit(x_train2, y_train)
y_pred4 = nb.predict(x_test2)
print("Accuracy Score Naive Bayes: ", accuracy_score(y_pred4, y_test))


#Load custom input image
img = cv.imread("/Users/arhan.sheth/Documents/Codes/DX/dx_imgClassification/customInput.jpeg")

#process image
img_arr = img.resize(img, (32, 32))
nx, ny, nrgb = img_arr.shape
img_arr2 = img.reshape(1, (nx * ny * nrgb))

#declare classes for comparision 
classes = ["airplane", "automobile", "bird", "cat", "deet", "frog", "horse", "ship", "truck"]
ans = model.predict(img_arr2)
#print the class of the predicted
print("Predicted Class: ", classes[ans[0]]) 
