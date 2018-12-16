import numpy as np
import math
import cv2
import imutils
from imutils import perspective
from imutils import contours


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def image_contour(img, flter_size, canny_threshold, canny_iteration_num):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (flter_size, flter_size), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, canny_threshold, canny_threshold)
    edged = cv2.dilate(edged, None, iterations=canny_iteration_num)
    edged = cv2.erode(edged, None, iterations=canny_iteration_num)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    (cnts, _) = contours.sort_contours(cnts)
    selected_contours = []
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) > 1000:
            selected_contours.append(c)

    if len(selected_contours) == 1:
        c = selected_contours[0]
        print np.shape(c)
        orig = img.copy()
        # cv2.drawContours(orig, [c], 0, (0, 255, 0), 3)
        # cv2.circle(orig, (c[0,0][0], c[0,0][1]), 5, (0, 0, 255), -1)
        # cv2.imshow('contour', orig)
        # cv2.waitKey(0)

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)

        box = np.array(box, dtype="int")

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

        (objX, objY) = midpoint((tlblX, tlblY), (trbrX, trbrY))
        # draw the  midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(objX), int(objY)), 5 * 2, (0, 255, 0), -1)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)
        # orientation = math.degrees(math.atan((tlblY - trbrY) / (trbrX - tlblX)))

        return [orig, c]
    else:
        return [None, None]

# def point_inside_contour(image, contour):
#
