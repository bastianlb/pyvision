import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
from visualize import render_camera_poses
from utils import load_camera_params, unfold_camera_param

DATA_DIR = "/home/ana/Downloads/bodytracking"
CAMERAS = ["cn01", "cn02", "cn03", "cn04"]


# invert the camera parameters for the reprojection into 3D
def calculate_reprojection_params(params):
    # extrinsic parameters: R = rotation matrix, T = translation vector
    # intrinsic parameters: f = focal length, c = principal point, k, p = distortion coefficients
    R, T, f, c, k, p = unfold_camera_param(params)
    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])

    K = K.reshape(3,3)
    K_inv = np.linalg.inv(K)
    R_t = R.T
    T_inv = (-R_t @ T).T
    return K_inv, R_t, T_inv


# convert a pixel coordinate into homogenous coordinates
def convert_to_hom_coords(point):
    point = np.append(point, 1)
    return point

# reproject one pixel into the corresponding 3D point
def reproject_pixel_in_3D(camera, px_coords):
    # convert the pixel into homogenous coordinates
    px_coords = convert_to_hom_coords(px_coords)
    # read the depth mask
    file_id = str(frame_id).zfill(10)
    fpath = os.path.join(DATA_DIR, camera, f"{file_id}_rgbd.tiff")
    depth_mask = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    # depth_mask is flipped
    # a pixel (x,y) in the color image can be accessed by (y,x) in the depth mask
    depth = depth_mask[px_coords[1]][px_coords[0]]/1000
    # the field of view of the depth camera is smaller than the one for the rgb images
    # need to check whether we have a measurement for the given pixel 
    if depth == 0.0:
        print(f"for camera:{camera} the pixel:{px_coords[0], px_coords[1]} has depth=0.0; cannot perform reprojection")
        return None
    # load the camera parameters (extrinsics and intrinsics)
    # they perform the mapping from a world point in 3D to a pixel in 2D
    params = load_camera_params(camera, DATA_DIR)
    # invert the camera parameters to get a reprojection from 2D into 3D
    K_inv, R_t, T_inv = calculate_reprojection_params(params)
    px_to_depth_cam = K_inv @ px_coords * depth
    depth_cam_to_world = R_t @ px_to_depth_cam + T_inv

    return depth_cam_to_world


# construct the 3D reprojections for each pixel
def construct_world_points(pixels):
    world_points = list()
    for cam in CAMERAS[:]:
        point = reproject_pixel_in_3D(cam, pixels[cam])
        if point is not None:
            world_points.append(point.reshape(3,))
    return world_points


# draw circles on the position of the pixels for a better intuition on where the 3D reprojections should end up
def draw_centers():
    for cam in CAMERAS[:]:
        file_id = str(frame_id).zfill(10)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
        image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("File not found: ", fpath)
        cv2.circle(image, (pixels[cam][0], pixels[cam][1]), 5, (255, 255, 255), 5)
        cv2.imwrite(cam + "_frame_1100_center.jpg", image)


if __name__ == "__main__":
    frame_id = 1100
    # center pixels of a person in the frame 1100
    pixels = {"cn01": np.array([680, 718]),
              "cn02": np.array([1695, 637]),
              "cn03": np.array([1692, 496]),
              "cn04": np.array([1473, 540])
              }
    # visualize the centers on the 2D images
    draw_centers()
    # reproject the pixels in 3D
    world_pts = np.array(construct_world_points(pixels))
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    # visualize the 3D points
    render_camera_poses(world_pts, vis, frame_id)
    app.add_window(vis)
    app.run()