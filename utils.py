import os.path as osp
import json
import open3d as o3d
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import torch

def unfold_camera_param(camera):
    world2color = np.linalg.inv(camera["color2world"])
    R, T = homogenous_to_rot_trans(world2color)
    camera["R"] = R
    camera["T"] = T
    fx = camera['fov_x']
    fy = camera['fov_y']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['c_x']], [camera['c_y']]]).reshape(-1, 1)
    k = camera['radial_params']
    p = camera['tangential_params']
    return R, T, f, c, k, p

def rot_trans_to_homogenous(rot, trans):
    """
    Args
        rot: 3x3 rotation matrix
        trans: 3x1 translation vector
    Returns
        4x4 homogenous matrix
    """
    X = np.zeros((4, 4))
    X[:3, :3] = rot
    X[:3, 3] = trans.T
    X[3, 3] = 1
    return X

def homogenous_to_rot_trans(X):
    """
    Args
        x: 4x4 homogenous matrix
    Returns
        rotation, translation: 3x3 rotation matrix, 3x1 translation vector
    """

    return X[:3, :3], X[:3, 3].reshape(3, 1)


def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap

def project_points_radial(x, R, T, K, k, p):
    """
    Project 3D points to 2D pixel coordinates with radial and tangential distortion.

    Args:
        x: Nx3 points in world coordinates.
        R: 3x3 Camera rotation matrix.
        T: 3x1 Camera translation vector.
        K: 3x3 Camera intrinsic matrix.
        k: 3x1 Camera radial distortion coefficients.
        p: 2x1 Camera tangential distortion coefficients.

    Returns:
        ypixel.T: Nx2 points in pixel space.
    """
    # Number of points
    n = x.shape[0]
    x = x
    # x = np.multiply([-1, 1, 1], x)
    # world2camera
    # https://www-users.cs.umn.edu/~hspark/CSci5980/Lec2_ProjectionMatrix.pdf
    xcam = R.dot(x.T) + T
    xcam = K @ xcam

    # perspective projection to map into pixels:
    # divide by the third component which represents the depth
    ypixel = xcam[:2] / (xcam[2] + 1e-5)
    # print(xcam[2])

    # r2 = np.sum(y**2, axis=0)
    # radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
    #                        np.array([r2, r2**2, r2**3]))
    # tan = p[0] * y[1] + p[1] * y[0]
    # y = y * np.tile(radial + 2 * tan,
    #                 (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    # ypixel = np.multiply(f, y) + c
    return ypixel.T

def project_points_opencv(x, R, T, K, k, p):
    k = np.array(k).reshape(-1, 1)
    p = np.array(p).reshape(-1, 1)
    #print(k.shape)
    #print(p.shape)
    dist_coefs = np.concatenate([k[0:2].T[0], p.T[0], k[2:].T[0]])
    # rvec, T perform a change of basis from world to camera coordinate system
    rvec = cv2.Rodrigues(R)[0]
    # project from 3D to 2D. projectPoints handles rotation and translation
    points_2d = cv2.projectPoints(x, rvec, T, K, dist_coefs)
    # TODO: why does projectPoints nest arrays like this?
    return np.array([x[0] for x in points_2d[0]])

def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)

    # Camera intrinsics after undistortion, run undistort.py before.
    K = camera['new_intrinsics']
    loc2d_opencv = project_points_opencv(x, R, T, K, k, p)
    loc2d = project_points_radial(x, R, T, K, k, p)

    print(camera["id"])
    print("------------")
    print(f" loc2d -> {loc2d} \n opencv -> {loc2d_opencv}")
    return loc2d

def load_rotation_matrix(rot: dict) -> np.ndarray:
    return Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()

def load_transform_matrix(trans: dict, rot: dict) -> np.ndarray:
    transform = np.zeros((4, 4), dtype=np.float32)
    transform[:3, :3] = load_rotation_matrix(rot)
    transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
    return transform

def extract_intrinsics_matrix(intrinsics_json: dict) -> np.ndarray:
    return np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                       [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                       [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

def load_cam_infos(take_path: Path) -> dict:
    camera_parameters = {}
    take_path = Path(take_path)
    camera_paths = sorted((take_path / "calibration").glob('camera*.json'))

    for cam_id, camera_path in enumerate(camera_paths, start=1):
        with camera_path.open() as f:
            cam_info = json.load(f)['value0']

        # Load intrinsics
        intrinsics = extract_intrinsics_matrix(cam_info['color_parameters']['intrinsics_matrix'])
        intrinsics[2, 1] = 0
        intrinsics[2, 2] = 1

        new_intrin = cam_info['color_parameters']['new_camera_matrix']

        intrinsics = np.array([
        [new_intrin[0][0], 0, new_intrin[0][2]],
        [0, new_intrin[1][1], new_intrin[1][2]],
        [0,  0,  1]
        ])

        # Load extrinsics
        extrinsics = load_transform_matrix(cam_info['camera_pose']['translation'], cam_info['camera_pose']['rotation'])
        depth_extrinsics = extrinsics.copy()

        # Debug: Original extrinsics
        #print(f"Camera {cam_id} - Original Extrinsics:\n", depth_extrinsics)

        # Load color-to-depth transform
        color2depth_transform = load_transform_matrix(cam_info['color2depth_transform']['translation'],
                                                      cam_info['color2depth_transform']['rotation'])

        # Debug: Validate color-to-depth transform
        #print(f"Camera {cam_id} - Color-to-Depth Transform:\n", color2depth_transform)

        # Sanity checks
        rotation_matrix = color2depth_transform[:3, :3]
        translation_vector = color2depth_transform[:3, 3]

        # Check rotation matrix properties
        orthogonality = np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3), atol=1e-6)
        determinant = np.linalg.det(rotation_matrix)
        #print(f"Camera {cam_id} - Rotation Orthogonality Check: {orthogonality}")
        #print(f"Camera {cam_id} - Rotation Determinant: {determinant}")

        # Check translation values
        #print(f"Camera {cam_id} - Translation Vector: {translation_vector}")

        # Validate inverse transform
        inv_transform = np.linalg.inv(color2depth_transform)
        consistency_check = np.allclose(inv_transform @ color2depth_transform, np.eye(4), atol=1e-6)
        #print(f"Camera {cam_id} - Inverse Transform Consistency Check: {consistency_check}")

        # Apply transformation to extrinsics
        #extrinsics = np.matmul(extrinsics, color2depth_transform)
        #print(f"Camera {cam_id} - After Color-to-Depth Transform:\n", extrinsics)

        # Additional transformations (flipping and swapping)
        YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
        YZ_SWAP = rotation_to_homogenous(-np.pi/2 * np.array([1, 0, 0]))
        XZ_FLIP = rotation_to_homogenous(np.pi* np.array([0, 1, 0]))
        XZ_SWAP = rotation_to_homogenous(-np.pi/2 * np.array([0, 1, 0]))
        XY_FLIP = rotation_to_homogenous(np.pi * np.array([0, 0, 1]))
        XY_SWAP = rotation_to_homogenous(-np.pi/2 * np.array([0, 0, 1]))

        ROTATE_45_Z = rotation_to_homogenous(-np.pi / 4 * np.array([0, 0, 1]))
        ROTATE_45_X = rotation_to_homogenous(-np.pi / 4 * np.array([1, 0, 0]))
        ROTATE_22_5_X = rotation_to_homogenous(-np.pi / 8 * np.array([1, 0, 0]))
        ROTATE_45_Y = rotation_to_homogenous(-np.pi / 4 * np.array([0, 1, 0]))

        extrinsics = extrinsics @ YZ_FLIP
        #print(f"Camera {cam_id} - After Additional Transformations:\n", extrinsics)

        # Compute color2world and depth2world
        depth2world = extrinsics
        color2world = depth2world @ color2depth_transform

        # Add camera information to dictionary
        color_params = cam_info['color_parameters']
        radial_params = tuple(color_params['radial_distortion'].values())
        tangential_params = tuple(color_params['tangential_distortion'].values())

        camera_parameters[f'camera0{cam_id}'] = {
            'intrinsics': intrinsics,
            'new_intrinsics': new_intrinsics,
            'extrinsics': extrinsics,
            'fov_x': color_params['fov_x'],
            'fov_y': color_params['fov_y'],
            'c_x': color_params['c_x'],
            'c_y': color_params['c_y'],
            'width': color_params['width'],
            'height': color_params['height'],
            'radial_params': radial_params,
            'tangential_params': tangential_params,
            'depth_extrinsics': depth_extrinsics,
            'depth2world': depth2world,
            'color2world': color2world
        }

    return camera_parameters

def project_3d_to_2d(points_3d, intrinsics, extrinsics):
    """
    Project a 3D point cloud to 2D using camera intrinsics and extrinsics.

    Parameters:
    - points_3d: The 3D point cloud to project (tensor).
    - intrinsics: The camera intrinsic matrix (tensor).
    - extrinsics: The camera extrinsic matrix (tensor).

    Returns:
    - 2D coordinates of the projected points (tensor).
    """
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    if not isinstance(extrinsics, torch.Tensor):
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
    
    ones = torch.ones((points_3d.shape[0], 1), device=points_3d.device)
    points_3d_homogeneous = torch.cat([points_3d, ones], dim=1)
    
    # Convert from world to camera coordinates
    points_3d_camera = torch.mm(torch.inverse(extrinsics), points_3d_homogeneous.t()).t()[:, :3]
    
    # Project to 2D
    points_2d = torch.mm(intrinsics, points_3d_camera.t()).t()
    points_2d = points_2d[:, :2] / points_2d[:, 2].unsqueeze(-1)
    
    return points_2d
