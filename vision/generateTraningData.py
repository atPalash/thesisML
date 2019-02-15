#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from objectSelection import objectSelectorOOP

if __name__ == "__main__":
    object_list = {'objA': 97, 'objB': 98, 'objC': 99, 'objD': 100}
    object_name = raw_input('Enter object name: ')

    if object_name not in object_list.keys():
        print 'No object with name found'
        exit(0)

    realsense_img_cols = 1280
    realsense_img_rows = 720
    image_padding = 10
    reference_pix = (40, 40)
    padding_around_reference_pix = 10
    camera = objectSelectorOOP.RealSenseCamera(None, realsense_img_cols, realsense_img_rows, image_padding, True,
                                               area_threshold=2000)
    camera.start_streaming()

    rospy.init_node('camera_node_for_generating_training_data', anonymous=True)
    ready_for_snap_sub = rospy.Subscriber('ready_for_snap', String, camera.take_snap, object_name)
    snap_taken_pub = rospy.Publisher('snap_taken', String, queue_size=1)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        if camera.snap_taken:
            snap_taken_pub.publish('ready')
        rate.sleep()
