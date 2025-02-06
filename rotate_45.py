import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap

#file_path = "/home/victorkawai/121224_fornero_take_6/ksv1capture/export/pointclouds_e57/pointcloud_000005_camera01.ply"
file_path = "/home/victorkawai/121224_fornero_take_6/ksv1capture/export/pointclouds_fused/pointcloud_000060_rotated.ply"

mesh = o3d.io.read_point_cloud(file_path)

rotation_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, -1.0]])

center = mesh.get_center()

#mesh.transform(rotation_matrix, center=center)
YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
mesh.transform(YZ_FLIP)

o3d.visualization.draw_geometries([mesh])
#output_path = "/home/victorkawai/121224_fornero_take_6/ksv1capture/export/pointclouds_e57/pointcloud_000005_camera01_rotated.ply"
output_path = "/home/victorkawai/121224_fornero_take_6/ksv1capture/export/pointclouds_fused/pointcloud_000060_rotated2.ply"
o3d.io.write_point_cloud(output_path, mesh)