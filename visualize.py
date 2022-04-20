import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
import pickle

from utils import load_camera_params
from utils import project_pose, homogenous_to_rot_trans

DATA_DIR = "/Users/tonywang/Documents/University.nosync/5th_Semester/CAP/holistic_or_export/"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]

import numpy as np
import json
import os
from glob import glob
from smplmodel import load_model
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
import pickle


def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes', 'expression']:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        # for smplx results
        outputs.append(data)
    return outputs

def read_smpl_all(path='smpl/'):
    results = sorted(glob(os.path.join(path, '*.json')))
    # result is list of dics
    datas = []
    for result in results:
        data = read_smpl(result)
        datas.append(data)
    return datas

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh
load_mesh = o3d.io.read_triangle_mesh

def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh


def create(vis):
    pass

def update_vis(vis, mesh, body_model, params, model):
    data = read_smpl_all()
    print("test")
    first_frame = data[0][0]

    Rh = first_frame['Rh']
    Th = first_frame['Th']
    poses = first_frame['poses']
    shapes = first_frame['shapes']
    vertices = body_model(poses, shapes, Rh, Th, return_verts=True, return_tensor=False)[0]
    mesh.vertices = Vector3dVector(vertices)
    vis.add_geometry("test", model)


# project a 3D point in the world to the 2D views
def project_to_views(points):
    # cam 1 [750, 640]
    points = points / 1000
    for cam in CAMERAS[:]:
        file_id = str(frame_id).zfill(10)
        params = load_camera_params(cam, DATA_DIR)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_color.jpg")
        color = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        # shape is in form: [height, width, channel]
        height, width, _ = color.shape
        # y-val <-> height x-val <-> width
        kinect_offset = np.array([0.5, 0.5])
        if color is None:
            print("File not found: ", fpath)
        points2d = project_pose(points, params)
        for loc2d in points2d:
            # x, y = np.int16(np.round(loc2d - kinect_offset))
            if 0 < int(loc2d[0]) < width and 0 < int(loc2d[1]) < height:
                print("Point present in image")
                print("\n")
                # azure kinect uses reverse indexing
                cv2.circle(color, (int(loc2d[0]), int(loc2d[1])), 2, (0, 0, 255), 2)
        cv2.imwrite(cam + "_test.jpg", color)


coco_joints_def = {0: 'nose',
                   1: 'Leye', 2: 'Reye', 3: 'Lear', 4: 'Rear',
                   5: 'Lsho', 6: 'Rsho',
                   7: 'Lelb', 8: 'Relb',
                   9: 'Lwri', 10: 'Rwri',
                   11: 'Lhip', 12: 'Rhip',
                   13: 'Lkne', 14: 'Rkne',
                   15: 'Lank', 16: 'Rank'}


# add a placeholder sphere
def add_mesh_sphere(point, vis, name):
    # create a sphere in the origin of the world coordinate system
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.7, 0.7, 0.1])
    # translate the sphere to the desired location in the world
    mesh_sphere.translate(point)
    vis.add_geometry(name, mesh_sphere)

def isInRange(points, point):
    for ps in points:
        ps = ps / 1000
        for p in ps:
            distance = np.linalg.norm(p - point)
            if distance <= 0.2:
                return True
    return False

# display a 3D point in the world
# the world consists of point clouds gathered from each camera
def render_camera_poses(points, vis, frame_id):
    # origin of the world coordinate system
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry("coordinate_frame", mesh_frame)
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(4)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        # each camera has its own extrinsics w.r.t. to the world and its own intrinsics
        params = load_camera_params(cam, DATA_DIR)
        # ply.transform(params["depth2world"])

        # plypoints = np.asarray(ply.points)
        # plycolors = np.asarray(ply.colors)

        # new_p = []
        # new_c = []
        # for p in range(len(plypoints)):
        #     if isInRange(points, plypoints[p]):
        #         new_p.append(plypoints[p])
        #         new_c.append(plycolors[p])

        # ply.points = o3d.utility.Vector3dVector(np.array(new_p))
        # ply.colors = o3d.utility.Vector3dVector(np.array(new_c))


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

    for i, joints in enumerate(points):
        joints = joints[:, :-2] / 1000
        for j, point in enumerate(joints):
            add_mesh_sphere(point, vis, f"{coco_joints_def[j]}_Person: {i}")


# display a 3D point in the coordinate system of a depth camera and also in the world coordinate system
def render_single_transform(point, vis):
    cam = "cn02"
    file_id = str(frame_id).zfill(4)
    # point cloud in world coordinate system
    ply = o3d.io.read_point_cloud(os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply"))
    vis.add_geometry(f"{cam}-ply-camera", ply)
    params = load_camera_params(cam, DATA_DIR)
    depth2world = params["depth2world"]  # this is actually world2depth
    # extrinsics of the depth camera
    R, T = homogenous_to_rot_trans(depth2world)
    # the given point is in world coordinates
    # we need to move it in the coordinate system of the desired depth camera
    pt = R @ point.T + T
    pt = pt.reshape(3, )
    # add the point in the depth coordinate system
    add_mesh_sphere(pt, vis, 'sphere-cam')
    ply_world = deepcopy(ply)
    # move the pointcloud in the depth coordinate system
    ply_world.transform(depth2world)
    vis.add_geometry(f"{cam}-ply-world", ply_world)
    # add the same point in the world coordinate system
    add_mesh_sphere(point.reshape(3, ), vis, 'sphere-world')

def render_smpl(vis, frame_id):
    data_id = (frame_id - 2000) // 5

    body_model = load_model(gender='neutral')

    data = read_smpl_all()
    for i in range(len(data[data_id])):
        frame = data[data_id][i]
        Rh = frame['Rh']
        Th = frame['Th']
        poses = frame['poses']
        shapes = frame['shapes']
        vertices = body_model(poses, shapes, Rh, Th, return_verts=True, return_tensor=False)[0]
        model = create_mesh(vertices=vertices, faces=body_model.faces)
        vis.add_geometry(f'Person {i}', model)


if __name__ == "__main__":
    # draw ball at point
    frame_id = 2150

    np.set_printoptions(suppress=True)
    # points = np.array([[0.265301, -0.963982, 0.005958]])
    # points = np.array([[0., 0., 0.], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    with open('pred_voxelpose.pkl', 'rb') as f:
        preds = pickle.load(f)

    points_ = preds[frame_id]
    # new extrinsics
    # point = np.array([0.249721, -0.005661, -0.974014])
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(points_, vis, frame_id)
    render_smpl(vis, frame_id)

    # render_single_transform(point, vis)
    # project_to_views(points_)
    app.add_window(vis)
    app.run()
