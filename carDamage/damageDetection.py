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

#train_path = "/Users/arhan.sheth/Documents/Codes/DX/dx_imgClassification/carDamage/data3a/training"
#test_path = "/Users/arhan.sheth/Documents/Codes/DX/dx_imgClassification/carDamage/data3a/validation"

#(x_train, y_train) = train_path.load_data()
#(x_test, y_test) = test_path.load_data()

train_ds = keras.utils.image_dataset_from_directory(
    directory='data3a/training',
    )

validation_ds = keras.utils.image_dataset_from_directory(
    directory='data3a/validation',
    )


for image_batch, labels_batch in train_ds:
  print("Image Batch Shape:", image_batch.shape)
  print("Labels Batch Shape:", labels_batch.shape)
  break

