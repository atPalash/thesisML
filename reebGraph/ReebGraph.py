import numpy as np
import cv2
from imutils import paths
import os

from helpers.helpers import midpoint, image_contour


class ReebGraph:
    def __init__(self, img_pth, flt_sz, cnny_thrsh, cnny_itr):
        self.image_path = img_pth
        self.filter_size = flt_sz
        self.canny_threshold = cnny_thrsh
        self.canny_iteration_num = cnny_itr

    def get_image_contour(self):
        image_paths = sorted(list(paths.list_images(self.image_path)))
        for image_path in image_paths:
            image_type = image_path.split(os.path.sep)[2].split('_')
            if image_type[1] == 'RGB.png':
                image_np = cv2.imread(image_path)
                [rect_image, contour] = image_contour(image_np, self.filter_size, self.canny_threshold, self.canny_iteration_num)
                if rect_image is None or contour is None:
                    print 'error in image'
                else:
                    # for c in contour:
                    #     cv2.circle(rect_image, (c[0][0], c[0][1]), 5, (255, 0, 0), 0)
                    # # determine the most extreme points along the contour
                    extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
                    extRight = tuple(contour[contour[:, :, 0].argmax()][0])
                    extTop = tuple(contour[contour[:, :, 1].argmin()][0])
                    extBot = tuple(contour[contour[:, :, 1].argmax()][0])
                    #
                    cv2.circle(rect_image, extLeft, 8, (0, 0, 255), -1)
                    cv2.circle(rect_image, extRight, 8, (0, 255, 0), -1)
                    # # cv2.circle(rect_image, extTop, 8, (255, 0, 0), -1)
                    # # cv2.circle(rect_image, extBot, 8, (255, 255, 0), -1)
                    cv2.imshow('rect_img', rect_image)
                    cv2.waitKey(0)
                    mask_image = np.zeros((np.shape(rect_image)[0], np.shape(rect_image)[1]))
                    for col in range(np.shape(mask_image)[1]):
                        for row in range(np.shape(mask_image)[0]):
                            dist = cv2.pointPolygonTest(contour, (col, row), False)
                            if dist == -1:
                                mask_image[row][col] = 0
                            else:
                                mask_image[row][col] = 1

                    cv2.imshow('rect_img', rect_image)
                    cv2.imshow('masked_img', mask_image)
                    cv2.waitKey(0)


if __name__ == "__main__":
    rg = ReebGraph('images/objA/', 3, 80, 5)
    rg.get_image_contour()
    # tte = cv2.imread('images/objA/objA1_RGB.png')
    # cv2.imshow('ii',
    # tte)