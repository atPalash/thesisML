import sys
import math
import numpy as np
import cv2
from imutils import paths
import os

from helpers.helpers import midpoint, image_contour


class ReebGraph:
    def __init__(self, img_pth=None, flt_sz=3, cnny_thrsh=80, cnny_itr=5, gripper_width=20, realtime_cam=False):
        self.image_path = img_pth
        self.realtime_camera = realtime_cam
        if realtime_cam:
            self.image_path = None
        self.filter_size = flt_sz
        self.canny_threshold = cnny_thrsh
        self.canny_iteration_num = cnny_itr
        self.gripper_width = gripper_width
        self.mask_image = None
        self.row_const_col = []
        self.col_const_row = []
        self.selected_gripper_pixel = []
        self.contour = None
        self.contour_xyz = None
        self.image_np = None
        self.gripping_points = []
        self.object_orientation = None

    def get_image_contour(self, rgb_image=None, contour_xyz=None):
        self.row_const_col = []
        self.col_const_row = []
        if self.realtime_camera:
            rect_image = rgb_image
            self.contour_xyz = contour_xyz
            self.contour = [[x[0][0]] for x in contour_xyz]
            self.contour = np.asarray(self.contour)
        else:
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

    def get_saved_image_contour(self):
        image_paths = sorted(list(paths.list_images(self.image_path)))
        for image_path in image_paths:
            image_type = image_path.split(os.path.sep)[-1].split('_')
            if image_type[1] == 'RGB.png':
                self.image_np = cv2.imread(image_path)
                self.get_image_contour()

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
            print 'row varies col constant'
            self.find_suitable_gripping_pts(True, False)
        else:
            print 'col varies row constant'
            self.find_suitable_gripping_pts(False, True)

    def find_suitable_gripping_pts(self, row_const_col=False, col_const_row=False):
        suitable_points = []
        if row_const_col:
            for itr in range(0, len(self.row_const_col) - 1, 2):
                pt_info = []
                mid_pt = (
                int((self.row_const_col[itr][0] + self.row_const_col[itr + 1][0]) / 2), self.row_const_col[itr][1])
                thickness = abs(self.row_const_col[itr][0] - self.row_const_col[itr + 1][0])
                curvature = 0
                for itr2 in range(0, np.shape(self.mask_image)[0] - 1):
                    if int(abs(self.mask_image[itr2][mid_pt[1]] - self.mask_image[itr2 + 1][mid_pt[1]])) is 1:
                        curvature = curvature + 1
                if curvature < 3 and thickness < self.gripper_width:
                    pt_info.append(mid_pt)
                    pt_info.append(thickness)
                    pt_info.append(((self.row_const_col[itr][0],self.row_const_col[itr][1]), (self.row_const_col[itr+1][0],self.row_const_col[itr+1][1])))
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
                if curvature < 3 and thickness < self.gripper_width:
                    pt_info.append(mid_pt)
                    pt_info.append(thickness)
                    pt_info.append(((self.col_const_row[itr][0],self.col_const_row[itr][1]), (self.col_const_row[itr+1][0]
                                                                                              ,self.col_const_row[itr+1][1])))
                    suitable_points.append(pt_info)
        selected_pts = []

        for pt in suitable_points:
            selected_pts.append(pt[0])
            # sel_pts.append(pt[0])
            cv2.circle(self.mask_image, (pt[0][1], pt[0][0]), 1, (0, 255, 255), -1)
        mid_pts = np.array([x[0] for x in suitable_points])
        x_cord_mid = mid_pts[:, 0]
        y_cord_mid = mid_pts[:, 1]
        print "polyfit data ", np.polyfit(x_cord_mid, y_cord_mid, 1), math.degrees(math.atan(np.polyfit(x_cord_mid, y_cord_mid, 1)[1])), \
            math.degrees(math.atan(np.polyfit(x_cord_mid, y_cord_mid, 1)[0]))
        self.object_orientation = math.degrees(math.atan(np.polyfit(x_cord_mid, y_cord_mid, 1)[0]))

        for itr in range(len(suitable_points)):
            nearest1 = self.contour[self.closest_pt(suitable_points[itr][2][0], self.contour)]
            suitable_points[itr].append((nearest1[0][0], nearest1[0][1]))
            nearest2 = self.contour[self.closest_pt(suitable_points[itr][2][1], self.contour)]
            orientation_weight = 0
            # if itr < len(suitable_points) - 1:
            #     orientation = abs(math.degrees(math.atan2(abs(nearest2[0][1] - nearest1[0][1]),
            #                                               abs(nearest2[0][0] - nearest1[0][0]))))
            #     diff_orientation = abs(orientation - self.object_orientation)
            #     if diff_orientation is 0:
            #         orientation_weight = 1
            #     else:
            #         orientation_weight = 1/diff_orientation
            distance_from_mid = itr - len(suitable_points)/2
            if distance_from_mid is 0:
                distance_weight = 1
            else:
                distance_weight = 1/abs(itr - len(suitable_points)/2)
            suitable_points[itr].append((nearest2[0][0], nearest2[0][1]))
            suitable_points[itr].append(orientation_weight + distance_weight)
        suitable_points.sort(key=lambda pt_: pt_[5])
        # for i in range(-1, -50, -1):
        #     self.gripping_points.append((suitable_points[i][3], suitable_points[i][4]))
        self.gripping_points.append((suitable_points[-1][3], suitable_points[-1][4]))
        cv2.imshow('masked_image_gripper', self.mask_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def closest_pt(self, pt, sel_pts):
        min = sys.maxint
        index = -1
        for i in range(len(sel_pts)):
            dist_2 = (sel_pts[i][0][0] - pt[1]) ** 2 + (sel_pts[i][0][1] - pt[0]) ** 2
            if dist_2 != 0 and dist_2 < min:
                min = dist_2
                index = i
        return index

    # def weight_calculator(self, mid_pts_mat):

if __name__ == "__main__":
    rg = ReebGraph('../objectSelection/images/training/objA', 3, 80, 5, 20)
    rg.get_saved_image_contour()
