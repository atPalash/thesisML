#!/usr/bin/env python
import math

import cv2
import numpy as np
import rospy
import geometry_msgs
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

from objectSelection import objectSelectorOOP
from objectSelection import objectIdentifier
from reebGraph import ReebGraph
from helpers import FrameTransformation
from std_msgs.msg import Float64MultiArray


class GripperData:
    def __init__(self):
        self.pose_goal = PoseStamped()
        self.gripper_set = False

    def set_gripper(self, position=None, orient=None):
        if position is not None and orientation is not None and not self.gripper_set:
            self.pose_goal.header.stamp = rospy.Time.now()
            self.pose_goal.header.frame_id = "/panda_link0"
            self.pose_goal.pose.orientation.x = orient['x']
            self.pose_goal.pose.orientation.y = orient['y']
            self.pose_goal.pose.orientation.z = np.deg2rad(orient['z'])
            self.pose_goal.pose.orientation.w = orient['w']

            self.pose_goal.pose.position.x = position['x']
            self.pose_goal.pose.position.y = position['y']
            self.pose_goal.pose.position.z = position['z']
            self.gripper_set = True
            print self.pose_goal


def detect_object_and_gripping_point(data, arg):
    z_buffer = 0.0
    tf_endEffector_to_camera = []
    tf_robotBaseframe_to_end_effector = []
    orientation_EF = 0
    if type(data) is PoseStamped:
        end_effector_RF_position = data.pose.position
        end_effector_RF_orientation = data.pose.orientation
        rotation_matrix_wrt_camera = FrameTransformation.quaternion_to_rotation_matrix(end_effector_RF_orientation)
        tranformation_matrix_wrt_camera = FrameTransformation.transformation_matrix(rotation_matrix_wrt_camera,
                                                                                    end_effector_RF_position, z_buffer)
    elif type(data) is Float64MultiArray:
        current_robot_pose = data.data
        tf_endEffector_to_camera = [ 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0.02, 0.05, -0.05, 1]
        tf_endEffector_to_camera = np.transpose(np.reshape(tf_endEffector_to_camera, (4, 4)))
        tf_robotBaseframe_to_end_effector = np.transpose(np.reshape(current_robot_pose, (4, 4)))
        orientation_EF = FrameTransformation.quaternion_msg_from_matrix(tf_robotBaseframe_to_end_effector)
    # print tf_robotBaseframe_to_end_effector
    object_location_wrt_camera = [[arg[0]['x']], [arg[0]['y']], [arg[0]['z'] + z_buffer], [1]]
    # print object_location_wrt_camera
    object_location_wrt_endEffector = np.dot(tf_endEffector_to_camera, object_location_wrt_camera)

    object_location_wrt_robot_baseframe = np.dot(tf_robotBaseframe_to_end_effector, object_location_wrt_endEffector)
    object_location_wrt_robot_baseframe = {'x': object_location_wrt_robot_baseframe[0][0],
                                           'y': object_location_wrt_robot_baseframe[1][0],
                                           'z': object_location_wrt_robot_baseframe[2][0]}
    # was used with moveit
    # object_quaternion_wrt_robot_baseframe = quaternion_from_euler(math.radians(179.0), math.radians(0.0),
    #                                                               math.radians((90 + arg[0]['o'])))

    object_quaternion_wrt_robot_baseframe = {'x': orientation_EF.x,
                                             'y': orientation_EF.y,
                                             'z': arg[0]['o'],
                                             'w': orientation_EF.w}
    arg[1].set_gripper(object_location_wrt_robot_baseframe, object_quaternion_wrt_robot_baseframe)


if __name__ == "__main__":
    object_list_keypress = {'objC': 97, 'objD': 98}
    object_name = raw_input('Enter object to grasp: ')
    # object_name = 'objC'

    if object_name not in object_list_keypress.keys():
        print 'No object with name found'
        exit(0)
    gripper_data = GripperData()

    object_list = {0: 'objC', 1: 'objD'}
    selected_model = '/home/palash/thesis/thesisML/objectSelection/models/weights_best_RGB_2class.hdf5'
    object_identifier = objectIdentifier.ObjectIdentfier(selected_model, 4, 3, object_list)
    reeb_graph = ReebGraph.ReebGraph(gripper_width=1000, realtime_cam=True)
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(object_list=None, realsense_image_cols=1280, realsense_image_rows=720,
                                               realsense_image_padding=10, realsense_camera=True, flt_sz=3,
                                               cnny_thrsh=30,
                                               cnny_itr=10, area_threshold=1000)
    camera.set_reference_pixel(reference_pix, padding_around_reference_pix)
    camera.start_streaming()  # got realsense camera object at 2
    camera.get_image_data()
    images = camera.detected_object_images
    entire_image = camera.padded_image
    cv2.imshow('entire image', entire_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
        [prediction_name, prediction_percentage] = object_identifier.predict(image_rgb).split(':')
        cv2.putText(image_rgb, prediction_name + ": " + str(prediction_percentage),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        # cv2.imshow('detected_obj', image_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # prediction_name = "objC"
        if object_name == prediction_name:
            reeb_graph.get_image_contour(entire_image, image_contour_xyz)
            gripping_points = reeb_graph.gripping_points

            orientation = reeb_graph.object_orientation
            for c_pts in gripping_points:
                for contour_pt in image_contour_xyz:
                    if contour_pt[0][0][0] == c_pts[0][0] and contour_pt[0][0][1] == c_pts[0][1]:
                        x_cord_1 = contour_pt[1][0]
                        y_cord_1 = contour_pt[1][1]
                        z_cord_1 = contour_pt[1][2]
                    if contour_pt[0][0][0] == c_pts[1][0] and contour_pt[0][0][1] == c_pts[1][1]:
                        x_cord_2 = contour_pt[1][0]
                        y_cord_2 = contour_pt[1][1]
                        z_cord_2 = contour_pt[1][2]
                cv2.circle(entire_image, c_pts[0], 1, (255, 255, 0), -1)
                cv2.circle(entire_image, c_pts[1], 1, (0, 255, 255), -1)
                cv2.putText(entire_image, "({o:.1f})".format(o=orientation), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
                cv2.putText(entire_image, "({x:.1f}, {y:.1f}, {z:.1f})".format(x=x_cord_1, y=y_cord_1, z=z_cord_1), c_pts[0],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 1)
                cv2.putText(entire_image, "({x:.1f}, {y:.1f}, {z:.1f})".format(x=x_cord_2, y=y_cord_2, z=z_cord_2), c_pts[1],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)

            cv2.imshow('entire image', entire_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            object_location = {'x': x_cord_1, 'y': y_cord_1, 'z': z_cord_1, 'o': orientation}
            print object_location
            rospy.init_node('camera_node_for_gripping', anonymous=True)
            gripping_point_pub = rospy.Publisher('reply_gripping_point', PoseStamped, queue_size=1)
            gripping_point_sub = rospy.Subscriber('franka_current_position', Float64MultiArray,
                                                  detect_object_and_gripping_point,
                                                  (object_location, gripper_data))

            rate = rospy.Rate(10)  # 10hz
            while not rospy.is_shutdown():
                if gripper_data.gripper_set:
                    # print gripper_data.pose_goal
                    gripping_point_pub.publish(gripper_data.pose_goal)
                    rate.sleep()
        else:
            continue

