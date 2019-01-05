import math

import numpy as np
import cv2
from imutils import paths
import os

from helpers.helpers import midpoint, image_contour


class ReebGraph:
    def __init__(self, img_pth, flt_sz, cnny_thrsh, cnny_itr, gripper_width=20):
        self.image_path = img_pth
        self.filter_size = flt_sz
        self.canny_threshold = cnny_thrsh
        self.canny_iteration_num = cnny_itr
        self.gripper_width = gripper_width
        self.mask_image = None
        self.row_const_col = []
        self.col_const_row = []
        self.selected_gripper_pixel = []
        self.contour = None
        self.image_np = None

    def get_image_contour(self):
        image_paths = sorted(list(paths.list_images(self.image_path)))
        for image_path in image_paths:
            image_type = image_path.split(os.path.sep)[2].split('_')
            if image_type[1] == 'RGB.png':
                self.image_np = cv2.imread(image_path)
                self.row_const_col = []
                self.col_const_row = []
                [rect_image, self.contour] = image_contour(self.image_np, self.filter_size, self.canny_threshold,
                                                           self.canny_iteration_num)
                if rect_image is None or self.contour is None:
                    print 'error in image'
                else:
                    self.mask_image = np.zeros((np.shape(rect_image)[0], np.shape(rect_image)[1]))
                    for col in range(np.shape(self.mask_image)[1]):
                        for row in range(np.shape(self.mask_image)[0]):
                            dist = cv2.pointPolygonTest(self.contour, (col, row), False)
                            if dist == -1:
                                self.mask_image[row][col] = 0
                            else:
                                self.mask_image[row][col] = 1

                    self.plot_mid_points()

    def plot_mid_points(self):
        for col in range(np.shape(self.mask_image)[1]):
            for row in range(np.shape(self.mask_image)[0] - 1):
                if (self.mask_image[row + 1][col] - self.mask_image[row][col]) == 1:
                    self.row_const_col.append([row, col, 1])
                if (self.mask_image[row + 1][col] - self.mask_image[row][col]) == -1:
                    self.row_const_col.append([row, col, -1])

        for row in range(np.shape(self.mask_image)[0]):
            for col in range(np.shape(self.mask_image)[1] - 1):
                if (self.mask_image[row][col + 1] - self.mask_image[row][col]) == 1:
                    self.col_const_row.append([row, col, 1])
                if (self.mask_image[row][col + 1] - self.mask_image[row][col]) == -1:
                    self.col_const_row.append([row, col, -1])

        if len(self.row_const_col) > len(self.col_const_row):
            # for itr in range(0, len(self.row_const_col) - 1, 2):
            #     cv2.circle(self.mask_image,
            #                (self.row_const_col[itr][1],
            #                 int((self.row_const_col[itr][0] + self.row_const_col[itr + 1][0]) / 2)
            #                 ), 1, (0, 255, 255), -1)
            print 'row varies col constant'
            self.find_suitable_gripping_pts(True, False)
        else:
            # for itr in range(0, len(self.col_const_row) - 1, 2):
            #     cv2.circle(self.mask_image, (int((self.col_const_row[itr][1] + self.col_const_row[itr + 1][1]) / 2),
            #                                  self.col_const_row[itr][0]), 1, (0, 255, 255), -1)
            print 'col varies row constant'
            self.find_suitable_gripping_pts(False, True)

    def find_suitable_gripping_pts(self, row_const_col = False, col_const_row = False):
        suitable_points = []
        if row_const_col:
            for itr in range(0, len(self.row_const_col) - 1, 2):
                pt_info = []
                mid_pt = (int((self.row_const_col[itr][0] + self.row_const_col[itr + 1][0]) / 2), self.row_const_col[itr][1])
                thickness = abs(self.row_const_col[itr][0] - self.row_const_col[itr + 1][0])
                curvature = 0
                for itr2 in range(0, np.shape(self.mask_image)[0] - 1):
                    if int(abs(self.mask_image[itr2][mid_pt[1]] - self.mask_image[itr2 + 1][mid_pt[1]])) is 1:
                        curvature = curvature + 1
                if curvature < 3:
                    pt_info.append(mid_pt)
                    pt_info.append(thickness)
                    suitable_points.append(pt_info)

        elif col_const_row:
            for itr in range(0, len(self.col_const_row) - 1, 2):
                pt_info = []
                mid_pt = (self.col_const_row[itr][0], int((self.col_const_row[itr][1] + self.col_const_row[itr + 1][1]) / 2))
                thickness = abs(self.col_const_row[itr][1] - self.col_const_row[itr + 1][1])
                curvature = 0
                for itr2 in range(0, np.shape(self.mask_image)[1] - 1):
                    if int(abs(self.mask_image[mid_pt[0]][itr2] - self.mask_image[mid_pt[0]][itr2 + 1])) is 1:
                        curvature = curvature + 1
                if curvature < 3:
                    pt_info.append(mid_pt)
                    pt_info.append(thickness)
                    suitable_points.append(pt_info)

        for pts in suitable_points:
            cv2.circle(self.mask_image, (pts[0][1], pts[0][0]), 1, (0, 255, 255), -1)
        print 'done'
        cv2.imshow('masked image', self.mask_image)
        cv2.waitKey(0)

if __name__ == "__main__":
    rg = ReebGraph('images/objA/', 3, 80, 5, 100)
    rg.get_image_contour()