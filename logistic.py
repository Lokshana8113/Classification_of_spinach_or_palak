from google.colab import drive
drive.mount('/content/drive')
image_directory ="/content/drive/MyDrive/palak(preprocessed)1/"
SIZE = 150
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import cv2
from PIL import Image
import numpy as np
Dry_leaf = os.listdir(image_directory + 'Dry_leaf/')
for i, image_name in enumerate(Dry_leaf):    #Remember enumerate method adds a counter and returns the enumerate object

    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Dry_leaf/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)
Fresh_leaf = os.listdir(image_directory + 'Fresh_leaf/')
for i, image_name in enumerate(Fresh_leaf):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Fresh_leaf/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)
dataset = np.array(dataset)
label = np.array(label)
print("Dataset size is ", dataset.shape)
print("Label size is ", label.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)
print("Train size is ", X_train.shape)
print("Test size is ", X_test.shape)
from keras.utils import normalize
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
INPUT_SHAPE = (SIZE, SIZE, 3)
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
