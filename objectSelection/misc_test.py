# import os
#
# from imutils import paths
#
# imagePaths = sorted(list(paths.list_images('images')))
#
# for imagePath in imagePaths:
#     test_arr = imagePath.split(os.path.sep)[2].split('_')
#     if len(imagePath.split(os.path.sep)[2].split('_')) == 1:
#         name = test_arr[0].split('.')[0] + '_RGB.png'
#         os.rename(imagePath, imagePath.split(os.path.sep)[0] + '/' + imagePath.split(os.path.sep)[1] + '/' + name)

import numpy as np
import h5py

data_to_write = np.random.random(size=(100,20))
# with h5py.File('name-of-file.h5', 'w') as hf:
#     hf.create_dataset("name-of-dataset",  data=data_to_write)

with h5py.File('images/objA/objA435_BGRD.h5', 'r') as hf:
    data = hf['BGRD'][:]
    print np.shape(data)