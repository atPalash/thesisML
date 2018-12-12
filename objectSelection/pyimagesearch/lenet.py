# import the necessary packages
from keras.models import Sequential
from keras import layers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.applications import VGG16

window_size = 3

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # # first set of CONV => RELU => POOL layers
        # model.add(Conv2D(3, (5, 5), padding="same",
        #                  input_shape=inputShape))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
        #
        # model.add(conv_base)

        # first set of CONV => RELU => POOL layers

        model.add(Conv2D(20, (window_size, window_size), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(window_size, window_size), strides=(2, 2)))
        # model.add(layers.Dropout(0.5))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (window_size, window_size), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(window_size, window_size), strides=(2, 2)))
        # # model.add(layers.Dropout(0.5))
        model.add(Conv2D(50, (window_size, window_size), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(window_size, window_size), strides=(2, 2)))

        model.add(Conv2D(50, (window_size, window_size), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(window_size, window_size), strides=(2, 2)))

        # model.add(Conv2D(50, (window_size, window_size), padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(window_size, window_size), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model