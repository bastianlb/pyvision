import os.path as osp
import json
import open3d as o3d

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


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
    rot_mat = Rotation.from_rotvec(vec).as_matrix()
    homogenous = np.identity(4)
    homogenous[:3, :3] = rot_mat
    return homogenous



def load_camera_params(cam, dataset_root):
    scaling = 1000
    ds = {"id": cam}
    
    # Load intrinsics from JSON
    intrinsics_path = osp.join(dataset_root, "calibration", f'{cam}.json')
    print(intrinsics_path)
    assert osp.exists(intrinsics_path)
    with open(intrinsics_path, 'r') as f:
        intrinsics_data = json.load(f)['value0']
    
    # Extract color intrinsics
    color_intrinsics = intrinsics_data["color_parameters"]["intrinsics_matrix"]
    ds['fx'] = color_intrinsics["m00"]
    ds['fy'] = color_intrinsics["m11"]
    ds['cx'] = color_intrinsics["m20"]
    ds['cy'] = color_intrinsics["m21"]

    radial_distortion = intrinsics_data["color_parameters"]["radial_distortion"]
    tangential_distortion = intrinsics_data["color_parameters"]["tangential_distortion"]
    # Extract distortion coefficients
    ds["k"] = np.array([radial_distortion[f"m{i}0"] for i in range(6)])

    ds["p"] = np.array([tangential_distortion["m00"], tangential_distortion["m10"]])

    # Extract Depth-to-Color Transformation
    depth2color_translation = np.array(
        [intrinsics_data["color2depth_transform"]["translation"][f"m{i}0"] for i in range(3)]
    )
    
    depth2color_rotation = intrinsics_data["color2depth_transform"]["rotation"]
    depth2color_rotation_matrix = Rotation.from_quat(
        [
            depth2color_rotation["x"],
            depth2color_rotation["y"],
            depth2color_rotation["z"],
            depth2color_rotation["w"],
        ]
    ).as_matrix()
    depth2color = rot_trans_to_homogenous(depth2color_rotation_matrix, depth2color_translation)
    ds["depth2color"] = depth2color

    # Extract Camera Pose (Extrinsics)
    camera_pose_translation = np.array(
        [intrinsics_data["camera_pose"]["translation"][f"m{i}0"] for i in range(3)]
    )
    camera_pose_rotation = intrinsics_data["camera_pose"]["rotation"]
    camera_pose_rotation_matrix = Rotation.from_quat(
        [
            camera_pose_rotation["x"],
            camera_pose_rotation["y"],
            camera_pose_rotation["z"],
            camera_pose_rotation["w"],
        ]
    ).as_matrix()
    camera_pose_homogeneous = rot_trans_to_homogenous(
        camera_pose_rotation_matrix, camera_pose_translation
    )
    ds["camera_pose"] = camera_pose_homogeneous

    # Derive Color-to-World and Depth-to-World Transformations
    depth2world = camera_pose_homogeneous
    color2world = depth2world @ np.linalg.inv(depth2color)

    ds["depth2world"] = depth2world
    ds["color2world"] = color2world

    return ds

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
    
    # Convert points from world to camera coordinates
    xcam = R.dot(x.T) + T  # Shape: (3, N)

    # Perspective projection to map into normalized camera coordinates
    #y = xcam[:2] / (xcam[2] + 1e-8)  # Shape: (2, N)
    y = np.zeros((2, n))
    y[0] = xcam[0] / (xcam[2] + 1e-8)
    y[1] = xcam[1] / (xcam[2] + 1e-8)
    # Compute radial distance squared (r^2)
    r2 = np.sum(y**2, axis=0)  # Shape: (N,)

    # Apply radial distortion
    radial = 1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3  # Shape: (N,)

    # Apply tangential distortion
    x_tangential = 2 * p[0] * y[0] * y[1] + p[1] * (r2 + 2 * y[0]**2)
    y_tangential = p[0] * (r2 + 2 * y[1]**2) + 2 * p[1] * y[0] * y[1]

    # Combine distortions
    y_distorted = np.vstack([
        y[0] * radial + x_tangential,
        y[1] * radial + y_tangential
    ])  # Shape: (2, N)

    # Transform to pixel coordinates using intrinsic matrix
    ypixel = K[:2, :2] @ y_distorted + K[:2, 2].reshape(-1, 1)  # Shape: (2, N)

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

    K = np.array([f[0][0], 0, c[0][0],
                  0, f[1][0], c[1][0],
                  0, 0, 1])
    K = K.reshape(3,3)
    loc2d_opencv = project_points_opencv(x, R, T, K, k, p)
    loc2d = project_points_radial(x, R, T, K, k, p)

    #print(camera["id"])
    #print("------------")
    #print(f" loc2d -> {loc2d} \n opencv -> {loc2d_opencv}")
    return loc2d











from pathlib import Path
from scipy.spatial.transform import Rotation
import json
import numpy as np


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

def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap

def apply_camera_pose_transformations(extrinsics, color2depth_transform):
    """
    Apply the camera pose transformation using rotation and translation from color2depth transform.
    
    Args:
        extrinsics (numpy.ndarray): The extrinsic matrix representing the camera pose.
        color2depth_transform (numpy.ndarray): The transformation from color to depth camera frame.
        
    Returns:
        numpy.ndarray: The updated extrinsics after applying the camera pose transformations.
    """
    # Extract rotation and translation from the color2depth transformation
    c2d_rotation = color2depth_transform[:3, :3]
    c2d_translation = color2depth_transform[:3, 3]

    # Extract the camera pose (extrinsics) rotation and translation
    dcp_rotation = extrinsics[:3, :3]
    dcp_translation = extrinsics[:3, 3]

    # Combine the rotations: dcp.rotation * c2d_tf.rotation
    combined_rotation = np.dot(dcp_rotation, c2d_rotation)

    # Combine the translations: (dcp.rotation * c2d_tf.translation) + dcp.translation
    combined_translation = np.dot(dcp_rotation, c2d_translation) + dcp_translation

    # Create a new homogeneous transformation matrix with combined rotation and translation
    combined_extrinsics = np.identity(4)
    combined_extrinsics[:3, :3] = combined_rotation
    combined_extrinsics[:3, 3] = combined_translation

    return combined_extrinsics


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

        extrinsics = extrinsics #YZ_SWAP @ YZ_FLIP @ extrinsics @ YZ_FLIP  # @ XZ_SWAP @ XY_SWAP
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

# Additional helper function to project 3D points to 2D using camera parameters
def project_to_2d(point_3d, camera):
    """
    Project a 3D point to 2D pixel coordinates with radial and tangential distortion.
    
    Args:
        point_3d: np.ndarray of shape (3,), 3D point in world coordinates.
        camera: dict with camera parameters:
            - "color2world": 4x4 extrinsic matrix.
            - "fov_x": Focal length in x direction.
            - "fov_y": Focal length in y direction.
            - "c_x": Principal point x-coordinate.
            - "c_y": Principal point y-coordinate.
            - "radial_params": Radial distortion coefficients (k1, k2, k3).
            - "tangential_params": Tangential distortion coefficients (p1, p2).

    Returns:
        np.ndarray of shape (2,), 2D point in pixel coordinates.
    """
    # Convert the point to homogeneous coordinates
    extrinsic_matrix = np.linalg.inv(camera["color2world"])
    fx = camera['fov_x']
    fy = camera['fov_y']
    c_x = camera['c_x']
    c_y = camera['c_y']
    k = camera['radial_params']
    p = camera['tangential_params']
    
    # Construct intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0, c_x],
        [0, fy, c_y],
        [0,  0,  1]
    ])
    
    # Convert point to homogeneous coordinates
    point_3d_hom = np.append(point_3d, 1)  # Shape: (4,)
    
    # Apply extrinsic matrix to convert to camera coordinates
    point_cam = np.dot(extrinsic_matrix, point_3d_hom)  # Shape: (4,)
    point_cam = point_cam[:3]  # Keep only x, y, z
    
    # Project to normalized camera coordinates
    x_n = point_cam[0] / (point_cam[2] + 1e-8)
    y_n = point_cam[1] / (point_cam[2] + 1e-8)
    
    # Compute radial distance squared (r^2)
    r2 = x_n**2 + y_n**2
    
    # Apply radial distortion
    radial_distortion = 1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3
    x_radial = x_n * radial_distortion
    y_radial = y_n * radial_distortion
    
    # Apply tangential distortion
    x_tangential = 2 * p[0] * x_n * y_n + p[1] * (r2 + 2 * x_n**2)
    y_tangential = p[0] * (r2 + 2 * y_n**2) + 2 * p[1] * x_n * y_n
    
    # Combine distortions
    x_distorted = x_radial + x_tangential
    y_distorted = y_radial + y_tangential
    
    # Convert to pixel coordinates using intrinsics
    point_img = intrinsic_matrix[:2, :2] @ [x_n, y_n] + intrinsic_matrix[:2, 2]  # Normalize by z
    
    # Return 2D point in integer pixel coordinates
    return point_img.astype(np.int32)