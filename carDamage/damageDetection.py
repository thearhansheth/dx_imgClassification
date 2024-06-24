#importing required packages onto virtual env
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import cv2 as cv

#Loading datasets from local directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='data3a/training',
    )

# Calculate the total number of samples in the dataset
total_samples = tf.data.experimental.cardinality(train_ds).numpy()
train_size = int(0.8 * total_samples)
# Split the dataset into training and testing sets
train_ds1 = train_ds.take(train_size)
test_ds = train_ds.skip(train_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='data3a/validation',
    )

#preprocessing for validation data
#Convert dataset into list to obtain all batches
validation_list = list(validation_ds)

#extract image and label
for image_batch, labels_batch in validation_list:
  validation_image = image_batch
  validation_label = labels_batch
  break

#converting all to numpy format for better adaptability
validation_image = validation_image.numpy()
validation_label = validation_label.numpy()

train_list = list(train_ds1)
test_list = list(test_ds)

#using variables to store batch data for easier access
for image_batch, labels_batch in train_list:
  train_image = image_batch
  #print("Image Batch Shape:", image_batch.shape)
  train_label = labels_batch
  #print("Labels Batch Shape:", labels_batch.shape)
  break

train_image = train_image.numpy()
train_label = train_label.numpy()

for image_batch, labels_batch in test_list:
  test_image = image_batch
  #print(image_batch.shape)
  test_label = labels_batch
  #print(labels_batch.shape)
  break

test_image = test_image.numpy()
test_label = test_label.numpy()

#Converting into 2d arrays for scikit-learn
# Converting train data
nsamples, nx, ny, nrgb = train_image.shape
train_image2 = train_image.reshape(nsamples, (nx * ny * nrgb))

# Converting test data
nsamples, nx, ny, nrgb = test_image.shape
test_image2 = test_image.reshape(nsamples, (nx * ny * nrgb))


#Converting Validation data into 2d Array
nsamples, nx, ny, nrgb = validation_image.shape
validation_image2 = validation_image.reshape(nsamples, (nx * ny * nrgb))

# Loading Random Forest Model
rF = RandomForestClassifier()
rF.fit(train_image2, train_label)
# Making prediction using Random Forest
rF_pred = rF.predict(test_image2)
print("Accuracy Score Random Forest:", accuracy_score(rF_pred, test_label))

# Loading KNN Model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_image2, train_label)
# Making prediction using KNN
knn_pred = knn.predict(test_image2)
print("Accuracy Score KNN:", accuracy_score(knn_pred, test_label)) 

# Loading Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(train_image2, train_label)
# Making prediction using Decision Tree
dt_pred = dt.predict(test_image2)
print("Accuracy Score Decision Tree:", accuracy_score(dt_pred, test_label))

# Loading Naive Bayes Model
nb = GaussianNB()
nb.fit(train_image2, train_label)
# Making prediction using Naive Bayes
nb_pred = nb.predict(test_image2)
print("Accuracy Score Naive Bayes:", accuracy_score(nb_pred, test_label))


#Making predictions for validation data
rF_val_pred = rF.predict(validation_image2)
print("Accuracy Report Random Forest (Validation):", accuracy_score(rF_val_pred, validation_label))

knn_val_pred = knn.predict(validation_image2)
print("Accuracy Report KNN (Validation):", accuracy_score(knn_val_pred, validation_label))

dt_val_pred = dt.predict(validation_image2)
print("Accuracy Report Decision Tree (Validation):", accuracy_score(dt_val_pred, validation_label))

nb_val_pred = nb.predict(validation_image2)
print("Accuracy Report Naive Bayes (Validation):", accuracy_score(nb_val_pred, validation_label))