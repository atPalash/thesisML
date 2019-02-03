import os
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import imutils
from imutils import perspective
from imutils import contours
import keyboard


class RealSenseCamera:
    def __init__(self, object_list=None, realsense_image_cols=848, realsense_image_rows=480, realsense_image_padding=10,
                 realsense_camera=True, flt_sz=3, cnny_thrsh=80, cnny_itr=5, area_threshold=1000):
        self.object_area_threshold = area_threshold
        self.realsense_present = realsense_camera
        self.realsense_img_cols = realsense_image_cols
        self.realsense_img_rows = realsense_image_rows
        self.realsense_image_padding = realsense_image_padding
        self.list_of_objects = object_list
        self.reference_pixel = (self.realsense_img_cols / 2, self.realsense_img_rows / 2)
        self.reference_pixel_padding = 10
        self.reference_pixel_depth = 0
        self.padded_image = None
        self.detected_object_images = []

        self.filter_size = flt_sz
        self.canny_threshold = cnny_thrsh
        self.canny_iteration_num = cnny_itr

        if self.realsense_present:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, self.realsense_img_cols, self.realsense_img_rows, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, self.realsense_img_cols, self.realsense_img_rows, rs.format.bgr8, 30)

            # Configure streaming instance parameters
            self.profile = None
            self.depth_sensor = None
            self.depth_scale = None
            self.align_to = None
            self.align = None
            self.frames = None
            self.aligned_frames = None
            self.aligned_depth_frame = None
            self.color_frame = None
            self.snap_taken = False

    def set_reference_pixel(self, ref_pixel, reference_pix_padding):
        self.reference_pixel = ref_pixel
        self.reference_pixel_padding = reference_pix_padding

    def get_reference_pixel_depth_from_camera(self, image, ref):
        start_pix_row = self.reference_pixel[1] - self.realsense_image_padding - self.reference_pixel_padding / 2
        end_pix_row = self.reference_pixel[1] - self.realsense_image_padding + self.reference_pixel_padding / 2
        start_pix_col = self.reference_pixel[0] - self.realsense_image_padding - self.reference_pixel_padding / 2
        end_pix_col = self.reference_pixel[0] -self.realsense_image_padding + self.reference_pixel_padding / 2
        image = image[start_pix_row: end_pix_row, start_pix_col: end_pix_col]
        data = self.get_image_depth_all_pixel(image, start_pix_row, start_pix_col, ref)
        reference_pixel_height = np.average(np.average(data))

        return reference_pixel_height

    def start_streaming(self):
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        for counter in range(20):
            self.pipeline.wait_for_frames()

        # self.get_image_data()

    # getting BGRD or XYZ data for a pixel or list of pixel
    def get_pixel_bgrd_or_xyz(self, image, pixel_coordinate_col=None, pixel_coordinate_row=None, contours=None,
                              bgrd=False, coordinate=False):
        # Get camera instrinsics
        depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # check data type passed
        if contours is None and pixel_coordinate_col is not None and pixel_coordinate_row is not None:
            # image padding is removed to receive correct distance
            depth = self.aligned_depth_frame.get_distance(int(pixel_coordinate_col) - self.realsense_image_padding,
                                                          int(pixel_coordinate_row) - self.realsense_image_padding)
            # data with XYZ values is returned wrt camera coordinates
            depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                                  [int(pixel_coordinate_col),
                                                                                   int(pixel_coordinate_row)], depth)
            if bgrd:
                bgrd = [image[pixel_coordinate_row][pixel_coordinate_col][0],  # B
                        image[pixel_coordinate_row][pixel_coordinate_col][1],  # G
                        image[pixel_coordinate_row][pixel_coordinate_col][2],  # R
                        depth_point_in_meters_camera_coords[2]]  # D
                return bgrd
            if coordinate:
                xyz = [depth_point_in_meters_camera_coords[0],  # X
                       depth_point_in_meters_camera_coords[1],  # Y
                       depth_point_in_meters_camera_coords[2]]  # Z
                return xyz
            return []
        elif contours is not None and pixel_coordinate_row is None and pixel_coordinate_col is None:
            pt_xyz = []
            pt_bgrd = []
            # image padding is removed to receive correct distance
            for pt in contours:
                depth = self.aligned_depth_frame.get_distance(int(pt[0][0]) - self.realsense_image_padding,
                                                              int(pt[0][1]) - self.realsense_image_padding)
                # data with XYZ values is returned wrt camera coordinates
                depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                                      [int(pt[0][0]),
                                                                                       int(pt[0][1])], depth)
                if bgrd:
                    bgrd = [image[pt[0][1]][pt[0][0]][0],  # B
                            image[pt[0][1]][pt[0][0]][1],  # G
                            image[pt[0][1]][pt[0][0]][2],  # R
                            depth_point_in_meters_camera_coords[2]]  # D
                    pt_bgrd.append([pt, bgrd])
                if coordinate:
                    xyz = [depth_point_in_meters_camera_coords[0],  # X
                           depth_point_in_meters_camera_coords[1],  # Y
                           depth_point_in_meters_camera_coords[2]]  # Z
                    pt_xyz.append([pt, xyz])

            if bgrd:
                return pt_bgrd
            elif coordinate:
                return pt_xyz
        else:
            return ['error']

    # getting BGRD data for a image
    def get_image_depth_all_pixel(self, image, pix_coordinate_col=None, pix_coordinate_row=None, ref=None):
        if self.realsense_present:
            depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            image_depth = np.zeros([np.shape(image)[0], np.shape(image)[1]])
            for pixel_coordinate_col in range(np.shape(image)[1]):
                for pixel_coordinate_row in range(np.shape(image)[0]):
                    # pixel_bgrd = []
                    # image padding is removed to receive correct distance
                    depth = self.aligned_depth_frame.get_distance(
                        int(pixel_coordinate_col) + pix_coordinate_col - self.realsense_image_padding,
                        int(pixel_coordinate_row) + pix_coordinate_row - self.realsense_image_padding)
                    depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                                          [int(
                                                                                              pixel_coordinate_col) + pix_coordinate_col,
                                                                                           int(
                                                                                               pixel_coordinate_row) + pix_coordinate_row],
                                                                                          depth)
                    if ref:
                        image_depth[pixel_coordinate_row, pixel_coordinate_col] = depth_point_in_meters_camera_coords[2]  # D
                    else:
                        image_depth[pixel_coordinate_row, pixel_coordinate_col] = self.reference_pixel_depth - depth_point_in_meters_camera_coords[2]  # D

            # if not ref:
            #     norm = np.amax(image_depth)
            #     image_depth[:, :] = image_depth[:, :] * (1000 / norm)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=0.5), cv2.COLORMAP_JET)
                # cv2.imshow('depth_image', depth_colormap)
                # cv2.waitKey(0)
            return image_depth
        else:
            return []


    @staticmethod
    def midpoint(ptA, ptB):
        return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

    def get_image_data(self, input_image=None):
        self.detected_object_images = []
        try:
            while True:
                if self.realsense_present:
                    # Wait for a coherent pair of frames: depth and color
                    self.frames = self.pipeline.wait_for_frames()
                    self.aligned_frames = self.align.process(self.frames)
                    self.aligned_depth_frame = self.aligned_frames.get_depth_frame()
                    self.color_frame = self.frames.get_color_frame()

                    if not self.aligned_depth_frame or not self.color_frame:
                        continue
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
                    color_image = np.asanyarray(self.color_frame.get_data())
                    image = color_image
                else:
                    image = cv2.imread(input_image)

                # creating border around the received image for neglecting objects at image extremes
                image_bordered = cv2.copyMakeBorder(image, self.realsense_image_padding, self.realsense_image_padding,
                                                    self.realsense_image_padding, self.realsense_image_padding,
                                                    cv2.BORDER_REPLICATE)

                self.padded_image = image_bordered.copy()
                # load the image, convert it to grayscale, and blur it slightly
                gray = cv2.cvtColor(image_bordered, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (self.filter_size, self.filter_size), 0)

                # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
                edged = cv2.Canny(gray, self.canny_threshold, self.canny_threshold)
                edged = cv2.dilate(edged, None, iterations=self.canny_iteration_num)
                edged = cv2.erode(edged, None, iterations=self.canny_iteration_num)

                # find contours in the edge map
                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]

                # sort the contours from left-to-right
                (cnts, _) = contours.sort_contours(cnts)

                for c in cnts:
                    area_ = cv2.contourArea(c)
                    # if the contour is not sufficiently large, ignore it
                    if cv2.contourArea(c) < self.object_area_threshold:
                        continue

                    # image copy to display results
                    orig = image_bordered.copy()

                    # draw lines on display image to indicate workspace boundary
                    cv2.line(orig, (self.realsense_image_padding, self.realsense_image_padding),
                             (self.realsense_image_padding, self.realsense_img_rows), (0, 0, 255), 20)
                    cv2.line(orig, (self.realsense_image_padding, self.realsense_image_padding),
                             (self.realsense_img_cols, self.realsense_image_padding), (0, 0, 255), 20)
                    cv2.line(orig, (self.realsense_img_cols, self.realsense_image_padding),
                             (self.realsense_img_cols, self.realsense_img_rows), (0, 0, 255), 20)
                    cv2.line(orig, (self.realsense_image_padding, self.realsense_img_rows),
                             (self.realsense_img_cols, self.realsense_img_rows), (0, 0, 255), 20)

                    # get the min area bounding box around the object/contour
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")

                    # get bounding box extremes around the object/contour
                    col_min = min(box[:, 0])
                    col_max = max(box[:, 0])
                    row_min = min(box[:, 1])
                    row_max = max(box[:, 1])
                    cv2.imshow('realsense_view', orig)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # neglect if object is outside the workspace boundary
                    if col_min < self.realsense_image_padding or col_max > self.realsense_img_cols \
                            or row_min < self.realsense_image_padding or row_max > self.realsense_img_rows:
                        print "object out of camera view"

                        continue
                    else:
                        # make a copy of RGB image(with border) and crop out the object area as defined by bounding box
                        # extremes
                        object_detected_img = image_bordered.copy()
                        object_detected_img = object_detected_img[row_min:row_max, col_min:col_max]

                        # make a copy of edged image(with border) and crop out the object area as defined by bounding
                        # box extremes
                        edge_detected_img = edged.copy()
                        edge_detected_img = edge_detected_img[row_min:row_max, col_min:col_max]

                        # receive image depth for each pixel in the selected object area
                        BGRD_detected_img = object_detected_img.copy()
                        BGRD_detected_img = self.get_image_depth_all_pixel(BGRD_detected_img, col_min, row_min, False)

                        if self.list_of_objects is None:
                            try:
                                if self.realsense_present:
                                    # Get reference pixel depth
                                    self.reference_pixel_depth = self.get_reference_pixel_depth_from_camera(orig, True)
                                    contour_xyz = self.get_pixel_bgrd_or_xyz(object_detected_img, None, None, c, False, True)
                                    object_dict = {'RGB': object_detected_img, 'EDGED': edge_detected_img,
                                                   'BGRD': BGRD_detected_img, 'contour': contour_xyz}
                                else:
                                    object_dict = {'RGB': object_detected_img, 'EDGED': edge_detected_img,
                                                   'BGRD': BGRD_detected_img, 'contour': c}
                                self.detected_object_images.append(object_dict)
                                # show original image of the workspace and detected object together
                                # try:
                                #     images = np.hstack(
                                #         (orig, cv2.resize(object_detected_img, (np.shape(orig)[1], np.shape(orig)[0]))))
                                #     # images = np.hstack((orig[:, :, 0], edged))
                                # except Exception as e:
                                #     print(e)
                                #     continue
                                #
                                # # Show images
                                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                                # cv2.imshow('RealSense', images)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()
                            except:
                                continue
                        else:
                            # order the points in the contour such that they appear in top-left, top-right, bottom-right
                            # and bottom-left order, then draw the outline of the rotated bounding box and draw contours
                            box = perspective.order_points(box)
                            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                            # loop over the points in the box and draw them, write pixel coordinate or world coordinate by
                            # changing the commented parts below
                            for (x, y) in box:
                                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                                # get XYZ coordinates in camera frame and write on image
                                xyz = self.get_pixel_bgrd_or_xyz(orig, int(x), int(y), None,  False, True)
                                cv2.putText(orig, "({x}, {y}, {z})".format(x=int(xyz[0] * 100), y=int(xyz[1] * 100),
                                                                           z=int(xyz[2] * 100)), (int(x), int(y)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                                # write image pixel coordinate
                                # cv2.putText(orig, "({x}, {y})".format(x=x, y=y), (int(x), int(y)),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

                            # plot reference pixel in the display image
                            cv2.circle(orig, (int(self.reference_pixel[0]), int(self.reference_pixel[1])), 5, (0, 255, 0),
                                       -1)

                            # get reference pixel camera coordiante and write in display image
                            xyz = self.get_pixel_bgrd_or_xyz(object_detected_img, self.reference_pixel[0],
                                                             self.reference_pixel[1], None, False, True)
                            cv2.putText(orig, "({x}, {y}, {z})".format(x=int(xyz[0] * 100), y=int(xyz[1] * 100),
                                                                       z=int(xyz[2] * 100)), (int(self.reference_pixel[0]),
                                                                                              int(self.reference_pixel[1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

                            # write reference pixel image coordinates in display image
                            # cv2.putText(orig, "({x}, {y})".format(x=int(self.reference_pixel[0]),
                            # y=int(self.reference_pixel[1])), (int(self.reference_pixel[0]), int(self.reference_pixel[1])),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

                            # unpack the ordered bounding box, then compute the midpoint between the top-left and top-right
                            # coordinates, followed by the midpoint between bottom-left and bottom-right coordinates
                            (tl, tr, br, bl) = box

                            (tltrX, tltrY) = self.midpoint(tl, tr)
                            (blbrX, blbrY) = self.midpoint(bl, br)

                            # compute the midpoint between the top-left and top-right points,
                            # followed by the midpoint between the top-righ and bottom-right
                            (tlblX, tlblY) = self.midpoint(tl, bl)
                            (trbrX, trbrY) = self.midpoint(tr, br)

                            (objX, objY) = self.midpoint((tlblX, tlblY), (trbrX, trbrY))
                            # draw the  midpoints on the image
                            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                            cv2.circle(orig, (int(objX), int(objY)), 5 * 2, (0, 255, 0), -1)

                            # get XYZ in world coordinate and plot in display image
                            xyz = self.get_pixel_bgrd_or_xyz(object_detected_img, objX, objY, None, False, True)
                            cv2.putText(orig, "({x}, {y}, {z})".format(x=int(xyz[0] * 100), y=int(xyz[1] * 100),
                                                                       z=int(xyz[2] * 100)), (int(objX), int(objY)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                            # draw lines between the midpoints
                            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                                     (255, 0, 255), 2)
                            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                                     (255, 0, 255), 2)
                            orientation = math.degrees(math.atan((tlblY - trbrY) / (trbrX - tlblX)))
                            # show original image of the workspace and detected object together
                            try:
                                images = np.hstack(
                                    (orig, cv2.resize(object_detected_img, (np.shape(orig)[1], np.shape(orig)[0]))))
                                # images = np.hstack((orig[:, :, 0], edged))
                            except Exception as e:
                                print(e)
                                continue

                            # Show images
                            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                            cv2.putText(images, "{:1f} sqpixel".format(cv2.contourArea(c)),
                                        (self.realsense_image_padding * 2, self.realsense_image_padding * 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                            cv2.imshow('RealSense', images)
                            key_press = cv2.waitKey(0)
                            cv2.imshow('edged', edge_detected_img)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                            self.write_data(key_press, object_detected_img, edge_detected_img, BGRD_detected_img)
                if self.list_of_objects is None:
                    break
        finally:
            # Stop streaming
            if self.realsense_present:
                print 'stopping camera'

    # save the images to their respective object types
    def write_data(self, key_press=None, object_detected_img=None, edge_detected_img=None, BGRD_detected_img=None, image_pth=None):
        if image_pth is None:
            image_path = 'images/sample_images/' + self.list_of_objects[key_press]
        else:
            image_path = image_pth
        try:
            file = os.listdir(image_path)
            filenum = len(file) / 3 + 1
            cv2.imwrite(image_path + image_path.split('/')[-1] + str(filenum) + '_RGB.png', object_detected_img)
            cv2.imwrite(image_path + image_path.split('/')[-1] + str(filenum) + '_EDGED.png', edge_detected_img)
            cv2.imwrite(image_path + image_path.split('/')[-1] + str(filenum) + '_DEPTH.png', BGRD_detected_img)
        except:
            if self.list_of_objects is not None:
                press_str = ""
                for key in self.list_of_objects:
                    press_str = press_str + list_of_objects[key] + ', '

                print 'This object is not in list, press ' + press_str

    def take_snap(self, data, object_type):
        self.get_image_data()
        images = self.detected_object_images

        if len(images) is 1:
            for img in images:
                image_rgb = img['RGB']
                image_depth = img['BGRD']
                image_edged = img['EDGED']
                self.write_data(None, image_rgb, image_edged, image_depth,
                                  image_pth='/home/palash/thesis/thesisML/objectSelection/images/sample_images/training_generator/'
                                            + object_type + '/')
        self.snap_taken = True


if __name__ == "__main__":
    realsense_img_cols = 1280
    realsense_img_rows = 720
    list_of_objects = {97: 'objA', 98: 'objB', 99: 'objC', 100: 'objD'}
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    realsense_present = True
    if realsense_present:
        camera = RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding,
                                 realsense_present, area_threshold=1000)
        camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
        camera.start_streaming()
        camera.get_image_data()
    else:
        test_image_path = '/home/palash/thesis/thesisML/objectSelection/images/4_Color.png'
        camera = RealSenseCamera(None, 1280, 720, image_padding, realsense_present)
        camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
        camera.get_image_data(test_image_path)

