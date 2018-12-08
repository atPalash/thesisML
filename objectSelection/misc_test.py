import os

from imutils import paths

imagePaths = sorted(list(paths.list_images('images')))

for imagePath in imagePaths:
    test_arr = imagePath.split(os.path.sep)[2].split('_')
    if len(imagePath.split(os.path.sep)[2].split('_')) == 1:
        name = test_arr[0].split('.')[0] + '_RGB.png'
        os.rename(imagePath, imagePath.split(os.path.sep)[0] + '/' + imagePath.split(os.path.sep)[1] + '/' + name)