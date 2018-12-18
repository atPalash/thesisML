import math

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
        self.row_const_col = []
        self.col_const_row = []
        self.gripper_rect = [6, 8]
        self.selected_gripper_pixel = []
        self.contour = None
        self.image_np = None

    def get_image_contour(self):
        image_paths = sorted(list(paths.list_images(self.image_path)))
        for image_path in image_paths:
            image_type = image_path.split(os.path.sep)[2].split('_')
            if image_type[1] == 'RGB.png':
                self.image_np = cv2.imread(image_path)
                [rect_image, self.contour] = image_contour(self.image_np, self.filter_size, self.canny_threshold,
                                                           self.canny_iteration_num)
                if rect_image is None or self.contour is None:
                    print 'error in image'
                else:
                    # for c in contour:
                    #     cv2.circle(rect_image, (c[0][0], c[0][1]), 5, (255, 0, 0), 0)
                    # # determine the most extreme points along the contour
                    extLeft = tuple(self.contour[self.contour[:, :, 0].argmin()][0])
                    extRight = tuple(self.contour[self.contour[:, :, 0].argmax()][0])
                    extTop = tuple(self.contour[self.contour[:, :, 1].argmin()][0])
                    extBot = tuple(self.contour[self.contour[:, :, 1].argmax()][0])
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
                            dist = cv2.pointPolygonTest(self.contour, (col, row), False)
                            if dist == -1:
                                self.mask_image[row][col] = 0
                            else:
                                self.mask_image[row][col] = 1
                    # cv2.imshow('masked_img', self.mask_image)
                    # cv2.waitKey(0)
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
            for itr in range(0, len(self.row_const_col) - 1, 2):
                cv2.circle(self.mask_image,
                           (self.row_const_col[itr][1],
                            int((self.row_const_col[itr][0] + self.row_const_col[itr + 1][0]) / 2)
                            ), 1, (0, 255, 255), -1)
            self.draw_rectangle(self.row_const_col)
        else:
            for itr in range(0, len(self.col_const_row) - 1, 2):
                cv2.circle(self.mask_image, (int((self.col_const_row[itr][1] + self.col_const_row[itr + 1][1]) / 2),
                                             self.col_const_row[itr][0]), 1, (0, 255, 255), -1)
            self.draw_rectangle(self.col_const_row)

            # orientation = math.degrees(math.atan((self.col_const_row[itr + 1][1] - self.col_const_row[itr][1]) /
            #                                      (self.col_const_row[itr + 1][1] - self.col_const_row[itr][1])))
            # orient = 180 - (orientation + 90 + math.degrees(math.atan(self.gripper_rect[0] / self.gripper_rect[1])))
            # pt1 = [self.col_const_row[itr][0] + np.sqrt(self.gripper_rect[0] ** 2 + self.gripper_rect[1] ** 2) * np.cos(
            #     orient),
            #        self.col_const_row[itr][1] + np.sqrt(self.gripper_rect[0] ** 2 + self.gripper_rect[1] ** 2) * np.sin(
            #            orient)]
            # pt2 = [self.col_const_row[itr][0] - np.sqrt(self.gripper_rect[0] ** 2 + self.gripper_rect[1] ** 2) * np.cos(
            #     orient),
            #        self.col_const_row[itr][1] - np.sqrt(self.gripper_rect[0] ** 2 + self.gripper_rect[1] ** 2) * np.sin(
            #            orient)]
            # cv2.rectangle(self.mask_image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 3)

    def draw_rectangle(self, pixel_array):
        pixel_array = np.asarray(pixel_array)
        pixel_array_x = np.subtract(pixel_array[0:len(pixel_array) - 1, 0], pixel_array[1:len(pixel_array), 0])
        pixel_array_y = np.subtract(pixel_array[0:len(pixel_array) - 1, 1], pixel_array[1:len(pixel_array), 1])
        orientation = np.arctan2(pixel_array_y, pixel_array_x) * 180/np.pi
        rect_half_diagonal = np.sqrt(self.gripper_rect[0]**2 + self.gripper_rect[1]**2)/2
        rect_angle_diagonal_wrt_x = 90 - np.arctan2(self.gripper_rect[1], self.gripper_rect[0]) * 180/np.pi

        orientation_1x = np.multiply(np.cos(orientation + rect_angle_diagonal_wrt_x), rect_half_diagonal)
        orientation_1y = np.multiply(np.sin(orientation + rect_angle_diagonal_wrt_x), rect_half_diagonal)
        orientation_2x = np.multiply(np.cos(180 + orientation + rect_angle_diagonal_wrt_x), rect_half_diagonal)
        orientation_2y = np.multiply(np.sin(180 + orientation + rect_angle_diagonal_wrt_x), rect_half_diagonal)

        rect_pts1_x = np.subtract(pixel_array[0: len(pixel_array) - 1, 0], orientation_1x)
        rect_pts1_y = np.subtract(pixel_array[0: len(pixel_array) - 1, 1], orientation_1y)

        rect_pts2_x = np.add(pixel_array[0: len(pixel_array) - 1, 0], orientation_2x)
        rect_pts2_y = np.add(pixel_array[0: len(pixel_array) - 1, 0], orientation_2y)
        cv2.imshow('mask_image', self.mask_image)
        cv2.waitKey(0)
        for itr in range (len(rect_pts1_x)):
            cv2.rectangle(self.image_np, (int(rect_pts1_x[itr]), int(rect_pts1_y[itr])), (int(rect_pts2_x[itr]),
                                                                                          int(rect_pts2_y[itr])),
                          (0, 255, 0), 2, 5)
            cv2.imshow('grasp_rectangle_img', self.image_np)
            cv2.waitKey(0)


if __name__ == "__main__":
    rg = ReebGraph('images/objA/', 3, 80, 5)
    rg.get_image_contour()
