# # import os
# #
# # from imutils import paths
# #
# # imagePaths = sorted(list(paths.list_images('images')))
# #
# # for imagePath in imagePaths:
# #     test_arr = imagePath.split(os.path.sep)[2].split('_')
# #     if len(imagePath.split(os.path.sep)[2].split('_')) == 1:
# #         name = test_arr[0].split('.')[0] + '_RGB.png'
# #         os.rename(imagePath, imagePath.split(os.path.sep)[0] + '/' + imagePath.split(os.path.sep)[1] + '/' + name)
# import cv2
#
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.cbook import get_sample_data
# from matplotlib._png import read_png
#
# data_to_write = np.random.random(size=(100, 20))
#
#
# # with h5py.File('name-of-file.h5', 'w') as hf:
# #     hf.create_dataset("name-of-dataset",  data=data_to_write)
#
# def myInt(myList):
#     return map(int, myList)
#
#
# def f(x):
#     return np.int(x)
#
#
# # with h5py.File('images/objA/objA252_BGRD.h5', 'r') as hf:
# #     data = hf['BGRD'][:]
# #     data = data[:,:,3]*100
# #     tt = np.amax(data)
# #     myMap = map(max, data)
# #     jj = list(map(max, data))
# #     kk = max(map(max, data))
# #     cv2.imshow('testy', cv2.imread('images/objA/objA252_RGB.png'))
# #     cv2.waitKey(0)
# #     f2 = np.vectorize(f)
# #     data = f2(data)
# #     tt = np.shape(data)[0]
# #     # data = cv2.cv.fromarray(data)
# #     # cv2.imshow('test', np.array(data, dtype = np.uint8 ))
# #     fig = plt.figure()
# #     ax = plt.gca(projection='3d')
# #
# #     zline = data
# #     for col in range(np.shape(zline)[0]):
# #         for row in range(np.shape(zline)[1]):
# #             if zline[col][row] >= 10:
# #                 zline[col][row] = 7
# #
# #     # zline = cv2.cvtColor(zline)
# #     zline = np.array(zline, dtype = np.uint8 )
# #     # zline = cv2.GaussianBlur(zline,(105,105),0)
# #     # zline = cv2.medianBlur(zline,5)
# #     xline, yine = np.mgrid[0:data.shape[0], 0:data.shape[1]]
# #     ax.plot_wireframe(xline, yine, zline, color='black')
# #     plt.show()
# x = np.linspace(-6, 6, 30)
# print x
import cv2

import h5py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate some sample data
import scipy.misc

with h5py.File('images/objA/objA848_BGRD.h5', 'r') as hf:
    data = hf['BGRD'][:]
    data = data[:,:,3]*100
lena = data

scale_factor = 255/10
for col in range(np.shape(lena)[0]):
    for row in range(np.shape(lena)[1]):
        if lena[col][row] >= 10:
            lena[col][row] = 1
        if 2 <= lena[col][row] < 2.5:
            lena[col][row] = 255
        if lena[col][row] < 2.5:
            lena[col][row] = 0

cv2.imshow('lena', lena)
cv2.waitKey(0)
# downscaling has a "smoothing" effect
lena = np.array(lena, dtype = np.uint8 )
lena = cv2.medianBlur(lena, 5)

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(xx, yy, lena, color='black')

# show it
plt.show()