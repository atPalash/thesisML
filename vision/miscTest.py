import math

import numpy as np
from geometry_msgs.msg import Quaternion
from tf.transformations import *
from geometry_msgs import *


def quat_to_mat(w, x, y, z):
    Nq = w * w + x * x + y * y + z * z
    if Nq < np.finfo(np.float).eps:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


# yaw (Z), pitch (Y), roll (X)
def euler_to_Quaternion(yaw, pitch, roll):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return [w, x, y, z]


def quaternion_msg_from_matrix(transformation):
    q = quaternion_from_matrix(transformation)
    msg = Quaternion()
    msg.x = q[0]

    msg.y = q[1]

    msg.z = q[2]

    msg.w = q[3]

    return msg


# # # wxyz
# Rotation_mat = quat_to_mat(0.00030684, -0.382530, 0.9239423, 0.000767)
# Transformation_mat = np.concatenate((Rotation_mat, [[-0.0001208], [-0.30696], [0.589525]]), 1)
# Transformation_mat = np.concatenate((Transformation_mat, [[0, 0, 0, 1]]), 0)

# camera_loc = [[0.10364757478237152], [-0.040704090148210526], [0.5360000133514404], [1]]
# robot_loc = np.dot(Transformation_mat, camera_loc)
# print robot_loc
# # RPY format for q_to_e and x, y, z,w for input quaternion
# q_to_e = [math.degrees(x) for x in euler_from_quaternion([-0.134, 0.99, 0.02, 0.027])]
# temp = [math.radians(x) for x in q_to_e]
#
# # x, y, z, w for output quaternion and RPY for input euler
# e_to_q = quaternion_from_euler(temp[0], temp[1], temp[2])
# e_to_q2 = quaternion_from_euler(math.radians(179.0), math.radians(0.0), math.radians((90 + 64)))
# print q_to_e
# print e_to_q
# print e_to_q2
#
# print quaternion_from_euler(math.radians(90/2),0,math.radians(180-58))
# print euler_to_Quaternion(math.pi/2,0,math.radians(-58))
# print euler_from_quaternion([0.618449525877659, -0.3428121700606599, -0.34281217006065984, 0.6184495258776589])
# t_mat = [-0.0006010787971238996, -0.9999865770700334, -0.0026891049154365507, 0.0,
#          -0.9999896394559501, 0.0006039051073349899, -0.0010503240398583088, 0.0,
#          0.0010519541590022193, 0.002688497489244088, -0.9999958326781656, 0.0,
#          1.5864708078526662e-05, -0.30659863321963776, 0.4859859621093075, 1.0]

# t_mat = [-0.0006010787971238996, -0.9999865770700336, -0.0026891049154366617, 0.0,
#          -0.9999896394559501, 0.0006039051073350454, -0.0010503240398583088, 0.0,
#          0.0010519541590022193, 0.0026884974892441996, -0.9999958326781656, 0.0,
#          1.5864708078526662e-05, -0.30659863321963776, 0.4859859621093075, 1.0]
t_mat_ef = [ 0, -1, 0, 0,
             1, 0, 0, 0,
             0, 0, 1, 0,
             0, 0.05, -0.06, 1]
# t_mat = np.transpose(np.reshape(t_mat, (4, 4)))
t_mat = [[2.85227356e-05, -9.99990311e-01,  3.53353635e-04, -1.99483929e-04],
 [-9.99989918e-01, -2.81857654e-05,  9.53611836e-04, -4.38307499e-01],
 [-9.53592637e-04, -3.53377272e-04, -9.99999483e-01,  4.50652443e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
t_mat_ef = np.transpose(np.reshape(t_mat_ef, (4,4)))
obj_loc_cam = [[0.05229568108916283], [-0.012006721459329128], [0.5350000262260437], [1]]
obj_loc_ef = np.dot(t_mat_ef, obj_loc_cam)
print obj_loc_ef
obj_loc_rob = np.dot(t_mat, obj_loc_ef)
print obj_loc_rob
# msg = quaternion_msg_from_matrix(t_mat)
# euler = euler_from_quaternion([msg.x, msg.y, msg.z, msg.w])
# print msg
# print euler
