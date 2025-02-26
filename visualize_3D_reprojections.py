import os
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
from visualize import render_camera_poses
from utils import unfold_camera_param, load_cam_infos

DATA_DIR = "./data"
CAMERAS = ["camera01", "camera02", "camera03", "camera04"]


# invert the camera parameters for the reprojection into 3D
def calculate_reprojection_params(params):
    # extrinsic parameters: R = rotation matrix, T = translation vector
    # intrinsic parameters: f = focal length, c = principal point,
    # k, p = distortion coefficients
    R, T, f, c, k, p = unfold_camera_param(params)
    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])

    K = K.reshape(3,3)
    K_inv = np.linalg.inv(K)
    R_t = R.T
    T_inv = (-R_t @ T).T
    return K_inv, R_t, T_inv


# convert a pixel coordinate into homogeneous coordinates
def convert_to_hom_coords(point):
    point = np.append(point, 1)
    return point

# reproject one pixel into the corresponding 3D point
def reproject_pixel_in_3D(camera, px_coords):
    # convert the pixel into homogeneous coordinates
    px_coords = convert_to_hom_coords(px_coords)
    print(px_coords)
    # read the depth mask
    file_id = str(frame_id).zfill(6)
    fpath = os.path.join(DATA_DIR, "depth", f"depth_{file_id}_{camera}.tiff")
    depth_mask = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    # depth_mask is flipped
    # a pixel (x,y) in the color image can be accessed by (y,x) in the depth mask
    print(depth_mask.shape)
    scaled_x = int(px_coords[0] * (depth_mask.shape[1] / 1920))
    scaled_y = int(px_coords[1] * (depth_mask.shape[0] / 1080))

# Ensure scaled coordinates are within bounds
    if 0 <= scaled_x < depth_mask.shape[1] and 0 <= scaled_y < depth_mask.shape[0]:
        depth = depth_mask[scaled_y][scaled_x] / 1000
    else:
        print("Scaled coordinates out of bounds:", scaled_x, scaled_y)
    # the field of view of the depth camera is smaller than the one for the rgb images
    # need to check whether we have a measurement for the given pixel
    if depth == 0.0:
        print(f"for camera:{camera} the pixel:{px_coords[0], px_coords[1]} has depth=0.0; cannot perform reprojection")
        return None
    # load the camera parameters (extrinsics and intrinsics)
    # they perform the mapping from a world point in 3D to a pixel in 2D
    params = load_cam_infos(DATA_DIR)[camera]
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
        file_id = str(frame_id).zfill(6)
        fpath = os.path.join(DATA_DIR, "color", f"color_{file_id}_{cam}.jpg")
        image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("File not found: ", fpath)
        cv2.circle(image, (pixels[cam][0], pixels[cam][1]), 5, (255, 255, 255), 5)
        print(image.shape) 
        cv2.imwrite(cam + "_frame_25_center.jpg", image)


if __name__ == "__main__":
    frame_id = 5
    # center pixels of a person in the frame 5
    pixels = {"camera01": np.array([320, 600]),
              "camera02": np.array([267, 171]),
              "camera03": np.array([1488, 552]),
              "camera04": np.array([799, 869])
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
