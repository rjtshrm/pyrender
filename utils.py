import numpy as np
from numpy import pi, cos, sin, arccos, arange
import math


def scale_vertices(xyz, scale=1.0, pad=0):
    """
    Scale xyz in unit sphere
    """
    m = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - m
    farthest_dist = np.amax(np.sqrt(np.sum(xyz ** 2, axis=-1)), axis=0, keepdims=True)
    xyz = np.divide(xyz, farthest_dist) * scale
    #np.savetxt(f"off/temp.xyz", xyz, delimiter=" ", newline="\n")
    return xyz


def read_off(file):
    off_data = open(file, "r")
    header = off_data.readline().strip()
    if header != "OFF":
        raise Exception("Error reading OFF file:: Invalid header")

    n_vert, n_face, _ = list(map(lambda a: int(a), off_data.readline().strip().split(' ')))

    vertices = []
    for i in range(n_vert):
        vert = list(map(lambda a: float(a), off_data.readline().strip().split(' ')))
        vertices.append(vert)

    faces = []
    for i in range(n_face):
        face = list(map(lambda a: int(a), off_data.readline().strip().split(' ')))[1:]
        faces.append(face)

    return np.array(vertices), np.array(faces)


def get_view_points(num_pts):
    # refer https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)

    xyz = np.column_stack([x, y, z])

    #pp.figure().add_subplot(111, projection='3d').scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.1)
    #pp.show()
    return xyz

def get_rotation(view):
    # refer https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
    longitude = - math.atan2(view[0], view[1])
    latitude = math.atan2(view[2], math.sqrt(view[0] ** 2 + view[1] ** 2))

    R_x = np.array(
        [[1, 0, 0], [0, math.cos(latitude), -math.sin(latitude)], [0, math.sin(latitude), math.cos(latitude)]])
    R_y = np.array(
        [[math.cos(longitude), 0, math.sin(longitude)], [0, 1, 0], [-math.sin(longitude), 0, math.cos(longitude)]])

    R = R_y.dot(R_x)
    return R

def rotation_matrix(roll=0, pitch=0, yaw=0):
    Rx = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    Ry = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    Rz = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def create_transformation_matrix(scale_matrix=np.eye(3), rot_matrix=np.eye(3), translation=np.zeros(3)):
    mat = np.eye(4)
    mat[0:3, 0:3] = scale_matrix @ rot_matrix
    mat[0:3, 3] = translation
    return mat

def xyz_2_homogeneous(xyz):
    return  np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

def homogeneous_2_xyz(xyzw):
    return  xyzw[:, 0:3]

def apply_transformation(transform_mat, xyzw):
    return (transform_mat @ xyzw.T).T
