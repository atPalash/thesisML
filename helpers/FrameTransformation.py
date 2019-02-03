import math
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion


def quaternion_to_rotation_matrix(orientation):
    qw = orientation.w
    qx = orientation.x
    qy = orientation.y
    qz = orientation.z
    Nq = qw * qw + qx * qx + qy * qy + qz * qz
    if Nq < np.finfo(np.float).eps:
        return np.eye(3)
    s = 2.0 / Nq
    qX = qx * s
    qY = qy * s
    qZ = qz * s
    wX = qw * qX
    wY = qw * qY
    wZ = qw * qZ
    xX = qx * qX
    xY = qx * qY
    xZ = qx * qZ
    yY = qy * qY
    yZ = qy * qZ
    zZ = qz * qZ
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])


def transformation_matrix(rotation_matrix, position):
    transformation_mat = np.concatenate((rotation_matrix, [[position.x], [position.y], [position.z]]), 1)
    transformation_mat = np.concatenate((transformation_mat, [[0, 0, 0, 1]]), 0)
    return transformation_mat


# yaw (Z), pitch (Y), roll (X)
def euler_to_quaternion(yaw, pitch, roll):
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


if __name__ == "__main__":
    # wxyz
    Rotation_mat = quaternion_to_rotation_matrix(0.0067, 0.92529, 0.37840, 0.024)
    Transformation_mat = np.concatenate((Rotation_mat, [[-0.0527], [0.4497], [0.6679]]), 1)
    Transformation_mat = np.concatenate((Transformation_mat, [[0, 0, 0, 1]]), 0)

    camera_loc = [[0.0545786581933], [-0.0828920602798], [0.401000022888], [1]]
    robot_loc = np.dot(Transformation_mat, camera_loc)
    print robot_loc
    # RPY format for q_to_e and x, y,z,w for input quaternion
    q_to_e = [math.degrees(x) for x in euler_from_quaternion([-0.134, 0.99, 0.02, 0.027])]
    temp = [math.radians(x) for x in q_to_e]

    # x, y, z, w for output quaternion and RPY for input euler
    e_to_q = quaternion_from_euler(temp[0], temp[1], temp[2])
    e_to_q2 = quaternion_from_euler(math.radians(179.0), math.radians(0.0), math.radians((90 + 64)))
    print q_to_e
    print e_to_q
    print e_to_q2
    #
    # print quaternion_from_euler(math.radians(90/2),0,math.radians(180-58))
    # print euler_to_Quaternion(math.pi/2,0,math.radians(-58))
    # print euler_from_quaternion([0.618449525877659, -0.3428121700606599, -0.34281217006065984, 0.6184495258776589])
