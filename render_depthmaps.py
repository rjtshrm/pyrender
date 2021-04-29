import matplotlib.pyplot as plt
import h5py
import pyrender
from utils import *
from scipy import ndimage

############ Camera Settings ##############
fov = 60
img_h = 640
img_w = 640
near_plane = 0.01
far_plane = 2

focal_length = 640#img_w / (2 * np.tan(fov * np.pi / 360))
cx = img_w / 2
cy = img_h / 2
intrinsics = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
###############################################

def render(off_file_path, depth_save_path, n_view_points=100):
    depth_maps = np.zeros((n_view_points, img_h, img_w))
    extr = np.zeros((n_view_points, 4, 4))
    intr = np.zeros((n_view_points, 3, 3))

    xyz_main, _faces = read_off(off_file_path)
    xyz_main = scale_vertices(xyz_main, scale=0.3).astype(np.float32)

    n_views = get_view_points(n_view_points).astype(np.float32)

    for i in range(n_view_points):
        # fix camera, rotate and translate object for particular camera view
        transform_mat = create_transformation_matrix(rot_matrix=get_rotation(n_views[i, ...]), translation=np.array([0, 0, 1]))
        tranform_xyz = homogeneous_2_xyz(apply_transformation(transform_mat, xyz_2_homogeneous(xyz_main)))
        depth = pyrender.render(np.ascontiguousarray(tranform_xyz), _faces,
                                   [focal_length, img_w, img_h, near_plane, far_plane, cx, cy], n_views[i, ...])


        # refer mesh fusion (https://github.com/davidstutz/mesh-fusion)
        # Dilation additionally enlarges thin structures
        depth = ndimage.morphology.grey_erosion(depth, size=(3, 3))

        depth_maps[i, ...] = depth
        extr[i, ...] = transform_mat
        intr[i, ...] = intrinsics

        #plt.imshow(depth, cmap='jet', aspect='auto')
        #plt.colorbar()
        #plt.show()

    with h5py.File(f'{depth_save_path}/{off_file_path.split("/")[-1]}.h5','w') as f:
        f.create_dataset("depth", shape=depth_maps.shape,data=depth_maps, compression='gzip', compression_opts=4)
        f.create_dataset("extrinsics", shape=extr.shape, data=extr, compression='gzip', compression_opts=4)
        f.create_dataset("intrinsics", shape=intr.shape, data=intr, compression='gzip', compression_opts=4)



if __name__ == "__main__":
    render("test.off", "data", 100)