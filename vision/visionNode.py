#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import cv2
from objectSelection import objectSelectorOOP
from objectSelection import objectIdentifier
from reebGraph import ReebGraph


class GripperData:
    def __init__(self):
        self.gripping_coord = None
        self.gripping_ready = False

    def set_gripper(self, grip_coord):
        self.gripping_coord = grip_coord
        self.gripping_ready = True


def detect_object_and_gripping_point(data, arg):
    # got realsense camera object at 2
    arg[2].start_streaming()
    arg[2].get_image_data()
    images = arg[2].detected_object_images
    entire_image = arg[2].padded_image

    x_cord_1 = 0
    y_cord_1 = 0
    z_cord_1 = 0
    x_cord_2 = 0
    y_cord_2 = 0
    z_cord_2 = 0
    orientation = 0
    for img in images:
        image_rgb = img['RGB']
        image_contour_xyz = img['contour']
        # got object_identifier object at 0
        if arg[0].predict(image_rgb) is arg[3]:
            arg[1].get_image_contour(entire_image, image_contour_xyz)
            gripping_points = arg[1].gripping_points
            orientation = arg[1].object_orientation
            for c_pts in gripping_points:
                for contour_pt in image_contour_xyz:
                    if contour_pt[0][0][0] == c_pts[0][0] and contour_pt[0][0][1] == c_pts[0][1]:
                        x_cord_1 = int(contour_pt[1][0] * 100)
                        y_cord_1 = int(contour_pt[1][1] * 100)
                        z_cord_1 = int(contour_pt[1][2] * 100)
                    if contour_pt[0][0][0] == c_pts[1][0] and contour_pt[0][0][1] == c_pts[1][1]:
                        x_cord_2 = int(contour_pt[1][0] * 100)
                        y_cord_2 = int(contour_pt[1][1] * 100)
                        z_cord_2 = int(contour_pt[1][2] * 100)
    arg[4].set_gripper([x_cord_1, y_cord_1, z_cord_1, orientation])


if __name__ == "__main__":
    object_list = {0: 'objA', 1: 'objB', 2: 'objC', 3: 'objD'}
    selected_model = '/home/palash/thesis/thesisML/objectSelection/models/weights_best_RGB.hdf5'
    object_identifier = objectIdentifier.ObjectIdentfier(selected_model, 4, 3, object_list)
    reeb_graph = ReebGraph.ReebGraph(gripper_width=1000, realtime_cam=True)
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(object_list=None, realsense_image_cols=848, realsense_image_rows=480,
                                               realsense_image_padding=10, realsense_camera=True, flt_sz=3,
                                               cnny_thrsh=30,
                                               cnny_itr=10, area_threshold=1000)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)

    object_list_keypress = {'objA': 97, 'objB': 98, 'objC': 99, 'objD': 100}
    object_name = raw_input('Enter object to grasp: ')

    if object_name not in object_list_keypress.keys():
        print 'No object with name found'
        exit(0)
    gripper_data = GripperData()
    rospy.init_node('camera_node_for_gripping_point', anonymous=True)
    gripping_point_sub = rospy.Subscriber('find_gripping_point', numpy_msg(Floats), detect_object_and_gripping_point,
                                          (object_identifier, reeb_graph, camera, object_name, gripper_data))
    gripping_point_pub = rospy.Publisher('reply_gripping_point', numpy_msg(Floats), queue_size=1)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        if gripper_data.gripping_ready:
            gripping_point_pub.publish(np.array(gripper_data.gripping_coord))
        rate.sleep()






        # cv2.circle(entire_image, c_pts[0], 1, (255, 255, 0), -1)
        # cv2.circle(entire_image, c_pts[1], 1, (0, 255, 255), -1)
        # cv2.putText(entire_image, "({o:.1f})".format(o=orientation), (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
        # cv2.putText(entire_image, "({x:.1f}, {y:.1f}, {z:.1f})".format(x=x_cord_1, y=y_cord_1, z=z_cord_1),
        #             c_pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
        # cv2.putText(entire_image, "({x:.1f}, {y:.1f}, {z:.1f})".format(x=x_cord_2, y=y_cord_2, z=z_cord_2),
        #             c_pts[1], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

        # cv2.imshow('detected_obj', image_rgb)
        # cv2.imshow('entire_image', entire_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
