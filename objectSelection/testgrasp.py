#!/usr/bin/env python2
import sys
from math import pi

import rospy
import moveit_commander
import geometry_msgs.msg

from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

PLACE_TARGET = (0, 0.5, 0.0)
CUBE = {'position': (+0.5,  0.0, 0.025), 'size': 0.05}


def moveto(pos):
    pose_target = Pose()

    # point downwards
    pose_target.orientation = Quaternion(*quaternion_from_euler(pi, 0, 0))

    # avoid moving below ground level/or the object
    pose_target.position = Point(*pos)
    pose_target.position.z += 0.15

    arm.set_pose_target(pose_target)
    arm.go()


def init_scene():
    rospy.sleep(1)

    # add table to scene
    TABLE_WIDTH = 1.0
    TABLE_LENGTH = 2.0
    PANDA_OFFSET = 0.145

    add_box('table', (TABLE_WIDTH / 2 - PANDA_OFFSET, 0, 0), (TABLE_WIDTH, TABLE_LENGTH, -0.001))
    add_box('cube', CUBE['position'], (CUBE['size'], CUBE['size'], CUBE['size']))


def add_box(name, pos, size):
    p = geometry_msgs.msg.PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.orientation.w = 1.0
    p.pose.position = Point(*pos)
    scene.add_box(name, p, size)

# -- main ----------------------------

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('picktest')

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

# hand and arm fail with 'no end-effector specified for pick action'
arm = moveit_commander.MoveGroupCommander('panda_arm')
arm_and_hand = moveit_commander.MoveGroupCommander('panda_arm_hand')
hand = moveit_commander.MoveGroupCommander('hand')

init_scene()

# this one fails with MoveIt "no sampler was constructed" error message
# moveto((0.2, 0, 0.5))
# moveto(CUBE['position'])
arm_and_hand.pick('cube')

moveit_commander.roscpp_shutdown()