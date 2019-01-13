from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2


class ObjectIdentfier:
    def __init__(self, model_path, num_of_classes, num_of_channels, object_list):
        self.model = load_model(model_path)
        self.classes = num_of_classes
        self.depth = num_of_channels
        self.IM_SIZE = 200
        self.list_of_objects = object_list

    def predict(self, image):
        image_BGRD = np.zeros((self.IM_SIZE, self.IM_SIZE, self.depth))
        image_rgb = image
        orig = image_rgb.copy()
        image_rgb = cv2.resize(image_rgb, (self.IM_SIZE, self.IM_SIZE))
        # image_rgb = img_to_array(image_rgb)
        image_BGRD[:, :, 0:3] = image_rgb
        image_BGRD = cv2.resize(image_BGRD, (self.IM_SIZE, self.IM_SIZE))
        image_BGRD = image_BGRD.astype("float") / 255.0
        image_BGRD = img_to_array(image_BGRD)
        image_BGRD = np.expand_dims(image_BGRD, axis=0)
        predictions = self.model.predict(image_BGRD)
        predictions = list(predictions[0])
        label = predictions.index(max(predictions))
        # build the label
        label = "{}: {:.2f}%".format(self.list_of_objects[label], predictions[label] * 100)
        return label
