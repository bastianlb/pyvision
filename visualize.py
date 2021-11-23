import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2

from utils import load_camera_params
from utils import project_pose, homogenous_to_rot_trans

DATA_DIR = "/data/develop/export_mkv_k4a/pointcloud_export"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]


def project_to_views(point):
    # cam 1 [750, 640]
    for cam in CAMERAS[:]:
        file_id = str(frame_id).zfill(10)
        params = load_camera_params(cam, DATA_DIR)
        loc2d = project_pose(point, params)[0]
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
        color = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if color is None:
            print("File not found: ", fpath)
        # shape is in form: [height, width, channel]
        height, width, _ = color.shape
        # y-val <-> height x-val <-> width
        kinect_offset = np.array([0.5, 0.5])
        x, y = np.int16(np.round(loc2d - kinect_offset))
        if 0 < x and x < width and 0 < y and y < height:
            print("Point present in image")
            print("\n")
            # azure kinect uses reverse indexing
            cv2.circle(color, (x, y), 1, (0, 255, 0), 1)
            cv2.imwrite(cam + "_test.jpg", color)


def add_mesh_sphere(point, vis, name):
    # add a placeholder sphere
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.7, 0.7, 0.1])
    mesh_sphere.translate(point)
    vis.add_geometry(name, mesh_sphere)


def render_camera_poses(points, vis, frame_id):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry("coordinate_frame", mesh_frame)
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(4)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        params = load_camera_params(cam, DATA_DIR)
        # ply.transform(params["depth2world"])
        vis.add_geometry(f"{cam}-ply", ply)

        print(f"Transforming for camera {cam}")
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        world2color = params["color2world"]
        camera_origin.transform(world2color)
        print(camera_origin.get_center())
        vis.add_geometry(f"{cam}-color", camera_origin)
        vis.add_3d_label(camera_origin.get_center(), cam)

        depth_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        world2depth = params["depth2world"]
        depth_origin.transform(world2depth)
        print(depth_origin.get_center())
        vis.add_geometry(f"{cam}-depth-cam", depth_origin)
        vis.add_3d_label(depth_origin.get_center(), cam + 'depth-')

    # add_mesh_sphere(point, vis, "sphere")
    # print(np.asarray(pcd.points))
    for i, point in enumerate(points):
        add_mesh_sphere(point, vis, f"sphere{i}")


def render_single_transform(point, vis):
    cam = "cn02"
    file_id = str(frame_id).zfill(4)
    ply = o3d.io.read_point_cloud(os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply"))
    vis.add_geometry(f"{cam}-ply-camera", ply)
    params = load_camera_params(cam, DATA_DIR)
    depth2world = params["depth2world"]
    R, T = homogenous_to_rot_trans(depth2world)
    add_mesh_sphere(R.T @ (point.T - T), vis, 'sphere-cam')
    # add the pointcloud in world coordinate system
    ply_world = deepcopy(ply)
    ply_world.transform(depth2world)
    vis.add_geometry(f"{cam}-ply-world", ply_world)
    add_mesh_sphere(point, vis, 'sphere-world')


if __name__ == "__main__":
    # draw ball at point
    frame_id = 1100
    np.set_printoptions(suppress=True)
    point = np.array([[0.265301, -0.963982, 0.005958], [0.265301, -0.963982, 0.005958]])

    # new extrinsics
    # point = np.array([0.249721, -0.005661, -0.974014])
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(point, vis, frame_id)
    # render_single_transform(point, vis)
    # project_to_views(point.reshape(1, 3))
    app.add_window(vis)
    app.run()
