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
model = load_model('objectDetector2.model')
print model.summary()
list_of_objects = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
for imagePath in imagePaths:
    image_type = imagePath.split(os.path.sep)[2].split('_')[1]
    if image_type == 'RGB.png':
        # load the image
        image = cv2.imread(imagePath)
        orig = image.copy()

        # pre-process the image for classification
        image = cv2.resize(image, (200, 200))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        predictions = model.predict(image)

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
