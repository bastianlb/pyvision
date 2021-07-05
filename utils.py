import os.path as osp
import json

import numpy as np
from scipy.spatial.transform import Rotation
import cv2

def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    # f = 0.5 * (camera['fx'] + camera['fy'])
    f = np.array([[fx], [fy]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
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

    return X[:3, :3], X[:3, 3].T

def invert_homogenous(X):
    rot = np.zeros((4, 4))
    rot[:3, :3] = X[:3, :3].T
    trans = np.identity(4)
    trans[:3, 3] = -X[:3, 3]
    rot[3, 3] = 1
    return rot @ trans

def load_extrinsics_for_cam(cam, dataset_root):
    ds = dict()
    intrinsics = osp.join(dataset_root, cam, 'camera_calibration.yml')
    assert osp.exists(intrinsics)
    fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
    color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
    ds['fx'] = color_intrinsics[0, 0]
    ds['fy'] = color_intrinsics[1, 1]
    ds['cx'] = color_intrinsics[0, 2]
    ds['cy'] = color_intrinsics[1, 2]
    # images are undistorted! Just put 0. Voxelpose assumes just 4 dist coeffs
    # dist = fs.getNode("color_distortion_coefficients").mat()
    # ds['k'] = np.array(dist[[0, 1, 4, 5, 6, 7]])
    # ds['p'] = np.array(dist[2:4])
    ds['k'] = np.zeros((3, 1))
    ds['p'] = np.zeros((2, 1))

    depth2color_r = fs.getNode('depth2color_rotation').mat()
    # depth2color_t is in mm by default, change all to meters
    depth2color_t = fs.getNode('depth2color_translation').mat() / 1000

    depth2color = rot_trans_to_homogenous(depth2color_r, depth2color_t.reshape(3))

    extrinsics = osp.join(dataset_root, cam, "world2camera.json")
    with open(extrinsics, 'r') as f:
        ext = json.load(f)
        trans = np.array([x for x in ext['translation'].values()])
        # NOTE: world2camera translation convention is in meters. Here we convert
        # to mm. Seems like Voxelpose was using mm as well.
        # trans = trans * 1000
        _R = ext['rotation']
        rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'], _R['w']]).as_matrix()
        ext_homo = rot_trans_to_homogenous(rot, trans)
        # flip coordinate transform back to opencv convention
        yz_transform = np.identity(4)
        yz_transform[1, 1] = -1.0
        yz_transform[2, 2] = -1.0
        ext_transformed = yz_transform @ ext_homo @ yz_transform.T

    # flip into voxelpose convention.. Z up
    YZ_FLIP = np.zeros((4, 4))
    YZ_FLIP[0, 0] = 1
    YZ_FLIP[1, 2] = -1
    YZ_FLIP[2, 1] = -1
    YZ_FLIP[3, 3] = 1

    # despite being called world2camera, it is a camera2world transform!
    world2camera = depth2color @ invert_homogenous(ext_transformed) @ YZ_FLIP.T
    R, T = homogenous_to_rot_trans(invert_homogenous(world2camera))
    ds["R"] = R
    ds["T"] = T.reshape(3, 1)
    ds["depth2color"] = depth2color
    # for debugging
    ds["id"] = cam
    return ds

def project_point_radial(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    n = x.shape[0]
    # some datasets use world2camera with subtraction
    # as opposed to addition
    xcam = R.dot(x.T + T)
    y = xcam[:2] / (xcam[2]+1e-5)
    # print(xcam[2])

    # r2 = np.sum(y**2, axis=0)
    # radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
    #                        np.array([r2, r2**2, r2**3]))
    # tan = p[0] * y[1] + p[1] * y[0]
    # y = y * np.tile(radial + 2 * tan,
    #                 (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)
    ypixel = np.multiply(f, y) + c
    return ypixel.T


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point_radial(x, R, T, f, c, k, p)
