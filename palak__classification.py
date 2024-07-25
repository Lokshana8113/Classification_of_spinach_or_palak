from google.colab import drive, files
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive')
image_directory ="/content/drive/MyDrive/palak(preprocessed)1/"
SIZE = 150
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

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
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train,
                         y_train,
                         batch_size = 64,
                         verbose = 1,
                         epochs = 10,
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )

model.save('models/palak_model_100epochs.h5')
label_categories = {0: "Fresh Leaf", 1: "Dry Leaf"}

# Get the index of the image to be loaded for testing from the user
n = int(input("Enter the index of the image to be loaded for testing: "))
img = X_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
predicted_prob = model.predict(input_img)[0][0]

# Apply threshold to the predicted probability
predicted_class = 1 if predicted_prob > 0.5 else 0

predicted_category = label_categories[predicted_class]
actual_category = label_categories[y_test[n]]

# Print the results
print("The predicted probability for this image is: ", predicted_prob)
print("The predicted class for this image is: ", predicted_class, "(", predicted_category, ")")
print("The actual label for this image is: ", y_test[n], "(", actual_category, ")")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get the model's predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

from keras.models import load_model
# load model
model = load_model('models/palak_model_100epochs.h5')
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
