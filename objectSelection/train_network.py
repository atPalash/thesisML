import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os


list_of_objects = {'objA': 0, 'objB': 1, 'objC': 2, 'objD': 3}
# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 30
INIT_LR = 1e-3
BS = 10
IM_SIZE = 200
# initialize the data and labels
print("[INFO] loading images...")
labels = []
data = []
classes = 4
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('images/training')))
# random.seed(42)
# random.shuffle(imagePaths)
count = 0
depth = 1
images = {}
image_BGRD = np.zeros((200, 200, depth))
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    label = imagePath.split(os.path.sep)[2]
    image = imagePath.split(os.path.sep)[3].split('_')

    # if image[1] == 'RGB.png':
    #     image_rgb = cv2.imread(imagePath)
    #     image_rgb = cv2.resize(image_rgb, (IM_SIZE, IM_SIZE))
        # image_BGRD[:, :, 0:3] = image_rgb
        # count = count + 1
    # if image[1] == 'DEPTH.png':
    #     image_depth = cv2.imread(imagePath)
    #     image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)
    #     image_depth = cv2.resize(image_depth, (IM_SIZE, IM_SIZE))
    #     image_depth = img_to_array(image_depth)
        # image_BGRD[:, :, 3] = image_depth
        # count = count + 1
    if image[1] == 'EDGED.png':
        image_edged = cv2.imread(imagePath)
        image_edged = cv2.cvtColor(image_edged, cv2.COLOR_BGR2GRAY)
        image_edged = cv2.resize(image_edged, (IM_SIZE, IM_SIZE))
        image_BGRD[:, :, 0] = image_edged
        count = count + 1
    if count == 1:
        data.append(image_BGRD)
        image_BGRD = np.zeros((200, 200, depth))
        # extract the class label from the image path and update the
        # labels list
        label = list_of_objects[label]
        labels.append(label)
        count = 0

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
# encoder = LabelEncoder()
# encoder.fit(labels)
# encoded_Y = encoder.transform(labels)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=classes)
testY = to_categorical(testY, num_classes=classes)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=90, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=IM_SIZE, height=IM_SIZE, depth=depth, classes=classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
print model.summary()
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# checkpoint best model
filepath = "models/weights_best_EDGED.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1, callbacks=callbacks_list)

# save the model to disk
print("[INFO] serializing network...")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
