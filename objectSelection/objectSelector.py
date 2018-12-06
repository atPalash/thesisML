import pyrealsense2 as rs
import numpy as np
import cv2
import math
import imutils
from imutils import perspective
from imutils import contours
from random import randint


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


realsense_img_cols = 848
realsense_img_rows = 480
remove_pixel = 0
list_of_objects = {97: 'objA', 98: 'objB', 99:'objC', 100:'objD'}
image_padding = 10


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, realsense_img_cols, realsense_img_rows, rs.format.z16, 30)
config.enable_stream(rs.stream.color, realsense_img_cols, realsense_img_rows, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

for i in range(10):
    pipeline.wait_for_frames()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        [rows, columns, channel] = np.shape(color_image)
        # image = color_image[10: 460, 10:620]
        image = color_image[remove_pixel: rows - remove_pixel, remove_pixel: columns - remove_pixel]
        image_bordered = cv2.copyMakeBorder(image, image_padding, image_padding, image_padding, image_padding,
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
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        # print cnts
        # print "'''''"
        # loop over the contours individually
        itr = 0
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 1000:
                continue

            orig = image_bordered.copy()
            cv2.line(orig, (image_padding, image_padding), (image_padding, realsense_img_rows),
                     (0, 0, 255), 20)
            cv2.line(orig, (image_padding, image_padding), (realsense_img_cols, image_padding),
                     (0, 0, 255), 20)
            cv2.line(orig, (realsense_img_cols, image_padding), (realsense_img_cols, realsense_img_rows),
                     (0, 0, 255), 20)
            cv2.line(orig, (image_padding, realsense_img_rows), (realsense_img_cols, realsense_img_rows),
                     (0, 0, 255), 20)
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

            box = np.array(box, dtype="int")
            col_min = min(box[:, 0])
            col_max = max(box[:, 0])
            row_min = min(box[:, 1])
            row_max = max(box[:, 1])
            if col_min < 10 or col_max > realsense_img_cols or row_min < 10 or row_max > realsense_img_rows:
                print "object out of camera view"
                continue
            else:
                col1 = col_min - image_padding
                col2 = col_max - image_padding
                row1 = row_min - image_padding
                row2 = row_max - image_padding
                object_detected_img = image.copy()
                object_detected_img = object_detected_img[row1:row2, col1:col2]

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box

                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

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
                    images = np.hstack((orig, cv2.resize(object_detected_img, (np.shape(orig)[1], np.shape(orig)[0]))))
                    # images = np.hstack((orig[:, :, 0], edged))
                except Exception as e:
                    print(e)
                    continue

                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.putText(images, "{:1f} sqpixel".format(cv2.contourArea(c)), (image_padding*2, image_padding*2), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
                cv2.imshow('RealSense', images)
                key_press = cv2.waitKey(0)
                try:
                    cv2.imwrite('images/' + list_of_objects[key_press] + '/' + list_of_objects[key_press] + str(randint(0,100)) + '.png', object_detected_img)
                except:
                    print 'this object is not in list, press A, B, C or D'
                    continue
finally:

    # Stop streaming
    pipeline.stop()