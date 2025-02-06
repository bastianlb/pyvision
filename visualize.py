import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
import torch

from utils import load_cam_infos
from utils import project_pose, homogenous_to_rot_trans, project_to_2d, project_3d_to_2d

DATA_DIR = "/home/victorkawai/121224_fornero_take_6/ksv1capture/export/"
CAMERAS = ["camera01", "camera02", "camera03", "camera04"]


# project a 3D point in the world to the 2D views
def project_to_views(point):
    # cam 1 [750, 640]
    for cam in CAMERAS[:]:
        file_id = str(frame_id).zfill(6)
        params = load_cam_infos(DATA_DIR)[cam]
        #print(params['intrinsics'])
        #loc2d = project_pose(point, params)
        intrinsics = params['new_intrinsics']
        extrinsics = params['color2world']
        torch_point = torch.tensor(point, dtype=torch.float32)
        loc2d = project_3d_to_2d(torch_point, intrinsics, extrinsics)
        print(loc2d)
        loc2d =loc2d[0]
        fpath = os.path.join(DATA_DIR, "color", f"color_{file_id}_{cam}_undistorted.jpg")
        color = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if color is None:
            print("File not found: ", fpath)
        # shape is in form: [height, width, channel]
        height, width, _ = color.shape
        # y-val <-> height x-val <-> width
        orbbec_offset = np.array([0.0, 0.0])
        #x, y = np.int16(np.round(loc2d - orbbec_offset))
        x, y = np.int16(loc2d)
        if 0 < x and x < width and 0 < y and y < height:
            print("Point present in image")

            # azure kinect uses reverse indexing
            cv2.circle(color, (x, y), 1, (0, 0, 255), 10)
            cv2.imwrite(cam + "_test.jpg", color)


# add a placeholder sphere
def add_mesh_sphere(point, vis, name):
    # create a sphere in the origin of the world coordinate system
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.7, 0.7, 0.1])
    # translate the sphere to the desired location in the world
    mesh_sphere.translate(point)
    vis.add_geometry(name, mesh_sphere)


# display a 3D point in the world
# the world consists of point clouds gathered from each camera
def render_camera_poses(points, vis, frame_id):
    # origin of the world coordinate system
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry("coordinate_frame", mesh_frame)
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(6)
#        fpath = os.path.join(DATA_DIR, "pointclouds_e57", f"pointcloud_{file_id}_{cam}.ply")
        fpath = os.path.join(DATA_DIR, "pointclouds_e57", f"pointcloud_{file_id}_{cam}.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        # each camera has its own extrinsics w.r.t. to the world and its own intrinsics
        params = load_cam_infos(DATA_DIR)[cam]
        # ply.transform(params["depth2world"])
        vis.add_geometry(f"{cam}-ply", ply)

        print(f"Transforming for camera {cam}")
        # create the origin of the camera -> in world coordinates
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # in order to display the camera origin in the correct location in the world coordinate system,
        # we need to apply the extrinsics of the camera.
        # the extrinsics can be seen as a mapping between two different coordinate frames.
        # in this case: we map from the world coordinate system to the camera coordinate system.
        world2color = params["color2world"]
        camera_origin.transform(world2color)
        print(camera_origin.get_center())
        vis.add_geometry(f"{cam}-color", camera_origin)
        vis.add_3d_label(camera_origin.get_center(), cam)

        # the depth and the color camera have their own coordinate system,
        # so we need to map the origin of the depth camera in world coordinates analogous to how we mapped the color one
        # by using the respective extrinsics
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


# display a 3D point in the coordinate system of a depth camera and also in the world coordinate system
def render_single_transform(point, vis):
    cam = "camera02"
    file_id = str(frame_id).zfill(6)
    # point cloud in world coordinate system
    ply = o3d.io.read_point_cloud(os.path.join(DATA_DIR, "pointclouds_e57", f"pointcloud_{file_id}_{cam}.ply"))
    vis.add_geometry(f"{cam}-ply-camera", ply)
    params = load_cam_infos(DATA_DIR)[cam]
    depth2world = params["depth2world"]  # this is actually world2depth
    # extrinsics of the depth camera
    R, T = homogenous_to_rot_trans(depth2world)
    # the given point is in world coordinates
    # we need to move it in the coordinate system of the desired depth camera
    pt = R @ point.T + T
    pt = pt.reshape(3,)
    # add the point in the depth coordinate system
    add_mesh_sphere(pt, vis, 'sphere-cam')
    ply_world = deepcopy(ply)
    # move the pointcloud in the depth coordinate system
    ply_world.transform(depth2world)
    vis.add_geometry(f"{cam}-ply-world", ply_world)
    # add the same point in the world coordinate system
    add_mesh_sphere(point.reshape(3,), vis, 'sphere-world')


if __name__ == "__main__":
    # draw ball at point
    frame_id = 5
    np.set_printoptions(suppress=True)
    
    #point = np.array([[-0.795794, -0.516614, -0.046237]])

    #point = np.array([[0.163907, 0.738747, -0.460022]])
    #point = np.array([[0.023019, -0.45939, -0.227529]])
    #point = np.array([[-0.755053, -0.522893, -0.044084]])

    #point = np.array([[-0.805985, -0.252734, 0.53880]])
    #point = np.array([[0.136309, -0.746423, 0.419467]])
    
    #point = np.array([[-0.764970, 0.162119, -0.601755]])
    point = np.array([[-0.418605, -1.447534, -0.467637]])
    #point = np.array([[0.009305, -0.458736, -0.221783]])
    # new extrinsics
    # point = np.array([0.249721, -0.005661, -0.974014])
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(point, vis, frame_id)
    #render_single_transform(point, vis)
    project_to_views(point.reshape(1, 3))
    app.add_window(vis)
    app.run()
