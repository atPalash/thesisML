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
        self.mask_image = None

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
                    # cv2.circle(rect_image, extLeft, 8, (0, 0, 255), -1)
                    # cv2.circle(rect_image, extRight, 8, (0, 255, 0), -1)
                    # # # cv2.circle(rect_image, extTop, 8, (255, 0, 0), -1)
                    # # # cv2.circle(rect_image, extBot, 8, (255, 255, 0), -1)
                    # cv2.imshow('rect_img', rect_image)
                    # cv2.waitKey(0)
                    self.mask_image = np.zeros((np.shape(rect_image)[0], np.shape(rect_image)[1]))
                    for col in range(np.shape(self.mask_image)[1]):
                        for row in range(np.shape(self.mask_image)[0]):
                            dist = cv2.pointPolygonTest(contour, (col, row), False)
                            if dist == -1:
                                self.mask_image[row][col] = 0
                            else:
                                self.mask_image[row][col] = 1
                    cv2.imshow('masked_img', self.mask_image)
                    cv2.waitKey(0)
                    self.plot_mid_points()

    def plot_mid_points(self):
        row_const_col = []
        for col in range(np.shape(self.mask_image)[1]):
            for row in range(np.shape(self.mask_image)[0]-1):
                if (self.mask_image[row+1][col] - self.mask_image[row][col]) == 1:
                    row_const_col.append([row, col, 1])
                if (self.mask_image[row+1][col] - self.mask_image[row][col]) == -1:
                    row_const_col.append([row, col, -1])

        col_const_row = []
        for row in range(np.shape(self.mask_image)[0]):
            for col in range(np.shape(self.mask_image)[1]-1):
                if (self.mask_image[row][col+1] - self.mask_image[row][col]) == 1:
                    col_const_row.append([row, col, 1])
                if (self.mask_image[row][col+1] - self.mask_image[row][col]) == -1:
                    col_const_row.append([row, col, -1])

        if len(row_const_col) > len(col_const_row):
            for itr in range(0, len(row_const_col) - 1, 2):
                cv2.circle(self.mask_image,
                           (row_const_col[itr][1], int((row_const_col[itr][0] + row_const_col[itr + 1][0]) / 2)
                            ), 1, (0, 255, 255), -1)
        else:
            for itr in range(0, len(col_const_row)-1, 2):
                cv2.circle(self.mask_image, (int((col_const_row[itr][1] + col_const_row[itr+1][1])/2), col_const_row[itr][0]
                                             ), 1, (0, 255, 255), -1)

        cv2.imshow('masked_img', self.mask_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    rg = ReebGraph('images/objA/', 3, 80, 5)
    rg.get_image_contour()
