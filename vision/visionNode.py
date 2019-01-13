from objectSelection import objectSelectorOOP
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
from imutils import paths
import cv2

imagePaths = sorted(list(paths.list_images('testImages')))
classes = 4

if __name__ == "__main__":
    print("[INFO] loading network...")
    # load the trained convolutional neural network
    model = load_model('../objectSelection/models/weights_best_RGB.hdf5')
    list_of_objects = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
    count = 0
    IM_SIZE = 200
    depth = 3
    image_BGRD = np.zeros((200, 200, depth))

    realsense_img_cols = 848
    realsense_img_rows = 480
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
    camera.start_streaming()
    images = camera.detected_object_images
    for img in images:
        image_rgb = img['RGB']
