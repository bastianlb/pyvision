import os

import numpy as np
import open3d as o3d
import cv2

from utils import load_extrinsics_for_cam, rot_trans_to_homogenous, invert_homogenous
from utils import project_pose

DATA_DIR = "/data/develop/export_mkv_k4a/test_system"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]

def project_to_views(point):
    frame_id = 25
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(10)
        params = load_extrinsics_for_cam(cam, DATA_DIR)
        loc2d = project_pose(point, params)[0]
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
        color = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if color is None:
            print("File not found: ", fpath)
        height, width, _ = color.shape
        x, y = loc2d
        if 0 < x and x < width and 0 < y and y < height:
            print(loc2d)
            cv2.circle(color, (int(x), int(y)), 5, (0, 255, 0), 2);
            cv2.imwrite(cam + "_test.jpg", color)


def render_camera_poses(point):
    # draw ball at point
    frame_id = 35

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    to_render = [mesh_frame]
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(4)
        ply = o3d.io.read_point_cloud(os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply"))
        params = load_extrinsics_for_cam(cam, DATA_DIR)
        A = rot_trans_to_homogenous(params["R"], params["T"])
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
        print(f"Transforming for camera {cam}")
        camera_origin.transform(A)
        print(camera_origin.get_center())
        to_render.append(ply)
        to_render.append(camera_origin)
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_sphere.translate(point)
    to_render.append(mesh_sphere)
    # print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries(to_render)

if __name__ == "__main__":
    point = [0.258, 0.973, 0.006]
    # render_camera_poses(point)
    project_to_views(np.array(point).reshape(1, 3))

