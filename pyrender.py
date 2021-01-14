import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from pytorch3d.renderer.cameras import look_at_rotation, look_at_view_transform

from numpy import pi, cos, sin, arccos, arange
import numpy as np
import pyrender
import math



def scale_vertices(xyz, scale=1.0):
    m = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - m
    farthest_dist = np.amax(np.sqrt(np.sum(xyz ** 2, axis=-1)), axis=0, keepdims=True)
    xyz = np.divide(xyz, farthest_dist) * scale
    #pp.figure().add_subplot(111, projection='3d').scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.1)
    #pp.show()
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


def get_view_points(num_pts, scale=1.0):
    # refer https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    indices = arange(0, num_pts, dtype=float) + 0.5

    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)

    xyz = np.column_stack([x, y, z]) * scale

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

def rot_matrix(roll=0, pitch=0, yaw=0):
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

def create_transformation_matrix(rot_matrix, translation):
    mat = np.eye(4)
    mat[0:3, 0:3] = rot_matrix
    mat[0:3, 3] = translation
    return mat

def xyz_2_homogeneous(xyz):
    return  np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

def homogeneous_2_xyz(xyzw):
    return  xyzw[:, 0:3]

def apply_transformation(transform_mat, xyzw):
    return (transform_mat @ xyzw.T).T


n_points = 100
fov = 45
img_h = 480
img_w = 480
near_plane = 0.01
far_plane = 100

focal_length = img_w / (2 * np.tan(fov * np.pi / 360))
cx = img_w / 2
cy = img_h / 2
im = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

xyz_main, _faces = read_off("test.off")
xyz_main = scale_vertices(xyz_main, scale=1.0).astype(np.float32)
#np.savetxt("off/main_vert.xyz", xyz_main, delimiter=" ", newline="\n")

n_views = get_view_points(n_points, scale=2).astype(np.float32)
#np.savetxt(f"off/n_views.xyz", n_views, delimiter=" ", newline="\n")


xl = np.arange(0, img_h)
yl = np.arange(0, img_w)

uu, vv = np.meshgrid(xl, yl, indexing='ij')

uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)

x = []
y = []
z = []

for i in range(n_points):
    transform_mat = create_transformation_matrix(get_rotation(n_views[i, ...]), np.array([0, 0, -3]))
    tranform_xyz = homogeneous_2_xyz(apply_transformation(transform_mat, xyz_2_homogeneous(xyz_main)))
    #np.savetxt(f"off/vert_{i}.xyz", tranform_xyz, delimiter=" ", newline="\n")
    depth = pyrender.render(np.ascontiguousarray(tranform_xyz), _faces, [focal_length, img_w, img_h, near_plane, far_plane, cx, cy], n_views[i, ...])

    d = depth
    u = uv[:, 0]
    v = uv[:, 1]
    w = d[u, v]
    valid = (w > near_plane) & (w < far_plane)
    u = u[valid]
    v = v[valid]
    w = w[valid]
    uvw = np.array([u, v, np.ones_like(u)]) * w
    xyz = (np.linalg.inv(im) @ uvw).T
    # transformation to match coordinate system b/w opengl and hz pinhole camera model
    xyz = xyz * np.array([1, 1, -1])
    fix_rot = rot_matrix(yaw=-np.pi/2)
    xyz = (fix_rot @ xyz.T).T
    #np.savetxt(f"off/tran_vert_{i}.xyz", xyz, delimiter=" ", newline="\n")
    xyz = homogeneous_2_xyz(apply_transformation(np.linalg.inv(transform_mat), xyz_2_homogeneous(xyz)))
    plt.imshow(depth, origin='lower')
    plt.show()

    x.extend(xyz[:, 0].ravel())
    y.extend(xyz[:, 1].ravel())
    z.extend(xyz[:, 2].ravel())

np.savetxt(f"off/out_vert.xyz", np.stack([x, y, z], axis=1), delimiter=" ", newline="\n")

