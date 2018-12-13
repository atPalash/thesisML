# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
import os

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
from imutils import paths
import cv2

imagePaths = sorted(list(paths.list_images('testImages')))
classes = 4
# load the trained convolutional neural network
print("[INFO] loading network..."
      "")
model = load_model('models/weights_best_RGB_DEPTH.hdf5')

print model.summary()
list_of_objects = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
count = 0
IM_SIZE = 200
depth = 5
images = {}
image_BGRD = np.zeros((200, 200, depth))
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    label = imagePath.split(os.path.sep)[1]
    image = imagePath.split(os.path.sep)[2].split('_')

    if image[1] == 'RGB.png':
        image_rgb = cv2.imread(imagePath)
        orig = image_rgb.copy()
        image_rgb = cv2.resize(image_rgb, (IM_SIZE, IM_SIZE))
        # image_rgb = img_to_array(image_rgb)
        image_BGRD[:, :, 0:3] = image_rgb
        count = count + 1
    if image[1] == 'DEPTH.png':
        image_depth = cv2.imread(imagePath)
        image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)
        image_depth = cv2.resize(image_depth, (IM_SIZE, IM_SIZE))
        # image_depth = img_to_array(image_depth)
        image_BGRD[:, :, 3] = image_depth
        count = count + 1
    if image[1] == 'EDGED.png':
        image_edged = cv2.imread(imagePath)
        image_edged = cv2.cvtColor(image_edged, cv2.COLOR_BGR2GRAY)
        image_edged = cv2.resize(image_edged, (IM_SIZE, IM_SIZE))
        # image_depth = img_to_array(image_depth)
        image_BGRD[:, :, 4] = image_edged
        count = count + 1
    if count == 3:
        image_BGRD = cv2.resize(image_BGRD, (200, 200))
        image_BGRD = image_BGRD.astype("float") / 255.0
        image_BGRD = img_to_array(image_BGRD)
        image_BGRD = np.expand_dims(image_BGRD, axis=0)


        # classify the input image
        predictions = model.predict(image_BGRD)

        predictions = list(predictions[0])
        label = predictions.index(max(predictions))
        # build the label

        label = "{}: {:.2f}%".format(list_of_objects[label], predictions[label] * 100)

        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Output", output)

        cv2.waitKey(0)
        count = 0

        image_BGRD = np.zeros((200, 200, depth))

