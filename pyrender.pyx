import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp
from pytorch3d.renderer.cameras import look_at_rotation, look_at_view_transform

from numpy import pi, cos, sin, arccos, arange
import numpy as np
cimport numpy as np

np.import_array()


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

def extrinsic(view):

    p = np.array((0, 0, 0))
    u = np.array((0, 1, 0))

    l = p - view
    l_norm = l / np.linalg.norm(l)

    s = np.cross(l, u)
    s_norm = s / np.linalg.norm(s)

    u_dash = np.cross(s_norm, l_norm)

    rot = np.stack([s_norm, u_dash, -l_norm], axis=0).T

    t = - rot @ view

    rt = np.concatenate([rot, np.expand_dims(t, axis=1)], axis=-1)
    rth = np.concatenate([rt, np.expand_dims([0, 0, 0, 1], axis=0)], axis=0)

    return rth

def tranform(xyz, ext):
    xyzw = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

    return (ext @ xyzw.T).T[:, 0:3]

# n_points = 100
# xyz, _faces = read_off("test.off")
# xyz = scale_vertices(xyz, scale=1.0).astype(np.float32)
# np.savetxt("off/main_vert.xyz", xyz, delimiter=" ", newline="\n")
#
# n_views = get_view_points(n_points, scale=2).astype(np.float32)
# np.savetxt(f"off/n_views.xyz", n_views, delimiter=" ", newline="\n")
# extrinsic_matrix = np.zeros((n_points, 4, 4), dtype=np.float32)
#
#
#
# for i, e_mat in enumerate(extrinsic_matrix):
#     extrinsic_matrix[i, ...] = extrinsic(n_views[i, ...])
#     xyz1 = tranform(xyz, extrinsic_matrix[i, ...])
#     np.savetxt(f"off/vert_{i}.xyz", xyz1, delimiter=" ", newline="\n")

cdef extern from "render.h":
    cdef struct CameraLoc:
        float* loc

    cdef struct Vertices:
        float* xyz
        int size

    cdef struct Faces:
        int* idxs
        int size

    cdef struct DepthBuffer:
        float* buffer

    cdef struct Camera:
        float focal_length
        int width
        int height
        float near
        float far
        float cx
        float cy

    int renderOFF(Vertices* vertices, Faces* faces, DepthBuffer* depthBuffer, Camera* camera, CameraLoc* cameraLoc)



def render(xyz, _faces, camera_init, camera_loc):

    focal_length, img_w, img_h, near_plane, far_plane, cx, cy = camera_init
    cdef Camera camera = [focal_length, img_w, img_h, near_plane, far_plane, cx, cy]


    depth_buffer = np.zeros((img_h, img_w), dtype=np.float32)
    cdef float[:, ::1] depth_buffer_view = depth_buffer
    cdef DepthBuffer depthBuffer = [&(depth_buffer_view[0, 0, 0])]


    cdef float[:, ::1] xyz_view = xyz.astype(np.float32)
    cdef Vertices vertices = [&(xyz_view[0, 0]), np.product(xyz.shape)]

    _faces = _faces.astype(np.int32)
    cdef int[:, ::1] faces_view = _faces.astype(np.int32)
    cdef Faces faces = [&(faces_view[0, 0]), np.product(_faces.shape)]

    cdef float[:] camera_loc_view = camera_loc.astype(np.float32)
    cdef CameraLoc cameraLoc = [&(camera_loc_view[0, 0])]

    renderOFF(&vertices, &faces, &depthBuffer, &camera, &cameraLoc)

    return depth_buffer