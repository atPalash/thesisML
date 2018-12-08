import pyrealsense2 as rs
import numpy as np
import cv2
import math
import imutils
from imutils import perspective
from imutils import contours
from random import randint


class realSenseCamera:
    def __init__(self, object_list, realsense_image_cols=848, realsense_image_rows=480, realsense_image_padding=10):
        self.realsense_img_cols = realsense_image_cols
        self.realsense_img_rows = realsense_image_rows
        self.realsense_image_padding = realsense_image_padding
        self.list_of_objects = object_list

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

    def start_streaming(self):
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        for counter in range(10):
            self.pipeline.wait_for_frames()

        self.get_streaming_data()

    # getting BGRD or XYZ data for a pixel
    def get_pixel_bgrd_or_xyz(self, image, pixel_coordinate_x, pixel_coordinate_y, bgrd, coordinate):
        depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        depth = self.aligned_depth_frame.get_distance(int(pixel_coordinate_x), int(pixel_coordinate_y))
        depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                              [int(pixel_coordinate_x),
                                                                               int(pixel_coordinate_y)], depth)
        if bgrd:
            bgrd = [image[pixel_coordinate_x][pixel_coordinate_y][0],  # B
                    image[pixel_coordinate_x][pixel_coordinate_y][1],  # G
                    image[pixel_coordinate_x][pixel_coordinate_y][2],  # R
                    depth_point_in_meters_camera_coords[2]]  # D
            return bgrd
        if coordinate:
            xyz = [depth_point_in_meters_camera_coords[0],  # X
                   depth_point_in_meters_camera_coords[1],  # Y
                   depth_point_in_meters_camera_coords[2]]  # Z
            return xyz

        return []

    # getting BGRD data for a image
    def get_image_bgrd(self, image):
        depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        image_bgrd = []
        for pixel_coordinate_x in range(np.shape(image)[0]):
            for pixel_coordinate_y in range(np.shape(image)[1]):
                depth = self.aligned_depth_frame.get_distance(int(pixel_coordinate_x), int(pixel_coordinate_y))
                depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin,
                                                                                      [int(pixel_coordinate_x),
                                                                                       int(pixel_coordinate_y)], depth)
                pixel_bgrd = [pixel_coordinate_x, pixel_coordinate_y,  # pixel_coordinates
                              image[pixel_coordinate_x][pixel_coordinate_y][0],  # B
                              image[pixel_coordinate_x][pixel_coordinate_y][1],  # G
                              image[pixel_coordinate_x][pixel_coordinate_y][2],  # R
                              depth_point_in_meters_camera_coords[2]]  # D
                image_bgrd.append(pixel_bgrd)

        return image_bgrd

    @staticmethod
    def midpoint(ptA, ptB):
        return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

    def get_streaming_data(self):
        try:
            while True:

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

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                [rows, columns, channel] = np.shape(color_image)
                num_rows = depth_image.shape[0]
                num_cols = depth_image.shape[1]

                image = color_image

                # creating border around the received image
                image_bordered = cv2.copyMakeBorder(image, self.realsense_image_padding, self.realsense_image_padding,
                                                    self.realsense_image_padding, self.realsense_image_padding,
                                                    cv2.BORDER_REPLICATE)

                # load the image, convert it to grayscale, and blur it slightly
                gray = cv2.cvtColor(image_bordered, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                # perform edge detection, then perform a dilation + erosion to
                # close gaps in between object edges
                edged = cv2.Canny(gray, 40, 40)
                edged = cv2.dilate(edged, None, iterations=5)
                edged = cv2.erode(edged, None, iterations=5)

                # find contours in the edge map
                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]

                # sort the contours from left-to-right and initialize the
                (cnts, _) = contours.sort_contours(cnts)

                for c in cnts:
                    # if the contour is not sufficiently large, ignore it
                    if cv2.contourArea(c) < 1000:
                        continue

                    orig = image_bordered.copy()
                    cv2.line(orig, (self.realsense_image_padding, self.realsense_image_padding),
                             (self.realsense_image_padding, self.realsense_img_rows), (0, 0, 255), 20)
                    cv2.line(orig, (self.realsense_image_padding, self.realsense_image_padding),
                             (self.realsense_img_cols, self.realsense_image_padding), (0, 0, 255), 20)
                    cv2.line(orig, (self.realsense_img_cols, self.realsense_image_padding),
                             (self.realsense_img_cols, self.realsense_img_rows), (0, 0, 255), 20)
                    cv2.line(orig, (self.realsense_image_padding, self.realsense_img_rows),
                             (self.realsense_img_cols, self.realsense_img_rows), (0, 0, 255), 20)
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

                    box = np.array(box, dtype="int")
                    col_min = min(box[:, 0])
                    col_max = max(box[:, 0])
                    row_min = min(box[:, 1])
                    row_max = max(box[:, 1])
                    if col_min < self.realsense_image_padding or col_max > self.realsense_img_cols \
                            or row_min < self.realsense_image_padding or row_max > self.realsense_img_rows:
                        print "object out of camera view"
                        continue
                    else:
                        object_detected_img = image_bordered.copy()
                        object_detected_img = object_detected_img[row_min:row_max, col_min:col_max]
                        edge_detected_img = edged.copy()
                        edge_detected_img = edge_detected_img[row_min:row_max, col_min:col_max]
                        BGRD_detected_img = object_detected_img.copy()
                        BGRD_detected_img = self.get_image_bgrd(BGRD_detected_img)

                        # order the points in the contour such that they appear
                        # in top-left, top-right, bottom-right, and bottom-left
                        # order, then draw the outline of the rotated bounding
                        # box
                        box = perspective.order_points(box)
                        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                        # loop over the original points and draw them
                        for (x, y) in box:
                            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                            xyz = self.get_pixel_bgrd_or_xyz(object_detected_img, x, y, False, True)
                            cv2.putText(orig,
                                        "({x}, {y}, {z})".format(x=int(xyz[0] * 100),
                                                                 y=int(xyz[1] * 100),
                                                                 z=int(xyz[2] * 100)),
                                        (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, (255, 255, 255), 2)
                        cv2.circle(orig, (int(self.realsense_img_cols / 2), int(self.realsense_img_rows / 2)), 5, (0, 0, 255), -1)
                        cv2.putText(orig, "({x}, {y})".format(x=0, y=0),
                                    (int(self.realsense_img_cols / 2), int(self.realsense_img_rows / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                        # unpack the ordered bounding box, then compute the midpoint
                        # between the top-left and top-right coordinates, followed by
                        # the midpoint between bottom-left and bottom-right coordinates
                        (tl, tr, br, bl) = box

                        (tltrX, tltrY) = self.midpoint(tl, tr)
                        (blbrX, blbrY) = self.midpoint(bl, br)

                        # compute the midpoint between the top-left and top-right points,
                        # followed by the midpoint between the top-righ and bottom-right
                        (tlblX, tlblY) = self.midpoint(tl, bl)
                        (trbrX, trbrY) = self.midpoint(tr, br)

                        # draw the  midpoints on the image
                        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                        # draw lines between the midpoints
                        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                                 (255, 0, 255), 2)
                        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                                 (255, 0, 255), 2)
                        orientation = math.degrees(math.atan((tlblY - trbrY) / (trbrX - tlblX)))

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
                        self.write_data(key_press, object_detected_img, edge_detected_img, BGRD_detected_img)

        finally:

            # Stop streaming
            self.pipeline.stop()

    def write_data(self, key_press, object_detected_img, edge_detected_img, BGRD_detected_img):
        try:
            random_int = str(randint(0, 1000))
            cv2.imwrite('images/' + self.list_of_objects[key_press] + '/' + self.list_of_objects[key_press] +
                        random_int + '_RGB.png', object_detected_img)
            cv2.imwrite('images/' + self.list_of_objects[key_press] + '/' + self.list_of_objects[key_press] +
                        random_int + '_EDGED.png', edge_detected_img)
            cv2.imwrite('images/' + self.list_of_objects[key_press] + '/' + self.list_of_objects[key_press] +
                        random_int + '_PCD.png', BGRD_detected_img)
        except:
            press_str = ""
            for key in self.list_of_objects:
                press_str = press_str + key + ', '

            print 'This object is not in list, you pressed' + key_press + ', press' + press_str