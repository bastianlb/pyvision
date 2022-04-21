import os
from copy import deepcopy

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
import pickle

from utils import load_camera_params, unfold_camera_param
from utils import project_pose, homogenous_to_rot_trans

import numpy as np
import json
import os
from glob import glob
from smplmodel import load_model
import open3d as o3d
import open3d.visualization.gui as gui
import cv2
import pickle
import trimesh
import open3d.visualization.rendering as rendering
import shutil
import pyrender

coco_joints_def = {0: 'Nose',
                   1: 'Leye', 2: 'Reye', 3: 'Lear', 4: 'Rear',
                   5: 'Lsho', 6: 'Rsho',
                   7: 'Lelb', 8: 'Relb',
                   9: 'Lwri', 10: 'Rwri',
                   11: 'Lhip', 12: 'Rhip',
                   13: 'Lkne', 14: 'Rkne',
                   15: 'Lank', 16: 'Rank'}

DATA_DIR = "data/holistic_or/"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]


# prepares output directories
def prepare_out_dirs(prefix:str='output/', dataDirs=['img']):
    result = []
    for dataDir in dataDirs:
        output_dir = os.path.join(prefix, dataDir)
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        print('Created', output_dir)
        os.makedirs(output_dir, exist_ok=True)
        result.append(output_dir)
    return result


# reads a json file
def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


# reads a smpl file
def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes', 'expression']:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        outputs.append(data)
    return outputs


def read_smpl_all(path='data/smpl_files/'):
    results = sorted(glob(os.path.join(path, '*.json')))
    # result is list of dicts
    datas = []
    for result in results:
        data = read_smpl(result)
        datas.append(data)
    return datas


# creates mesh out of vertices and faces
def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    # mesh.remove_vertices_by_index(list(range(600, 6890)))

    return mesh


# project a 3D point in the world to the 2D views
def project_to_views(points, frame_id):
    output_dir = prepare_out_dirs(dataDirs=['vertices_on_img'])[0]
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
                cv2.circle(color, (int(loc2d[0]), int(loc2d[1])), 1, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(output_dir, cam + f'Frame_.{frame_id}.jpg'), color)


# add a placeholder sphere
def add_mesh_sphere(point, vis, name):
    # create a sphere in the origin of the world coordinate system
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
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
        file_id = str(frame_id).zfill(4)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)

        # each camera has its own extrinsics w.r.t. to the world and its own intrinsics
        params = load_camera_params(cam, DATA_DIR)

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

    # will also print key points
    for i, joints in enumerate(points):
        joints = joints[:, :-2] / 1000
        for j, point in enumerate(joints):
            add_mesh_sphere(point, vis, f"{coco_joints_def[j]}_Person: {i}")


def render_smpl(frame_id, vis=None):
    # translates the frame_id to index
    data_id = (frame_id - 2000) // 5

    # loads the smpl model
    body_model = load_model(gender='neutral')

    # reads all smpl parameters
    data = read_smpl_all()
    # for debugging purposes we only visualize one person
    frame = data[data_id][1]
    Rh = frame['Rh']
    Th = frame['Th']
    poses = frame['poses']
    shapes = frame['shapes']

    # gets the vertices
    vertices = body_model(poses, shapes, Rh, Th, return_verts=True, return_tensor=False)[0]
    # the mesh 
    model = create_mesh(vertices=vertices, faces=body_model.faces)
    
    # projects vertices to 2d image
    project_to_views(np.asarray(model.vertices), frame_id)
    # project smpl onto 2d image with open3d render
    project_smpl(frame_id, model)
    # project smpl onto 2d image with pyrender
    project_smpl_pyrender(frame_id, model)

    # project to 3D space
    if vis != None:
        for i in range(len(data[data_id])):
            frame = data[data_id][i]
            Rh = frame['Rh']
            Th = frame['Th']
            poses = frame['poses']
            shapes = frame['shapes']
            vertices = body_model(poses, shapes, Rh, Th, return_verts=True, return_tensor=False)[0]
            model = create_mesh(vertices=vertices, faces=body_model.faces)
            vis.add_geometry(f'Person {i}', model)


# WARNING: Bugged
# rendering for camera 01 with pyrender
# From: https://github.com/alon1samuel/Visualize3DHumanModels
def project_smpl_pyrender(frame_id, mesh):
    output_dir = prepare_out_dirs(dataDirs=['pyrender_on_img'])[0]
    img = cv2.imread(os.path.join(DATA_DIR, 'cn01', '0000002150_color.jpg'), -1)


    fx = 1.02595850e+03
    fy = 1.02400476e+03
    cx = 1.02028540e+03
    cy = 7.76271240e+02

    R = np.array(
        [[-0.66923718,  0.74202016, -0.03908604],
        [ 0.50586555,  0.4164543 , -0.75542432],
        [-0.54426251, -0.52533031, -0.65406911]]
    )

    T = np.array(
        [[-0.22217233],
        [-0.29406874],
        [3.90767329]]
    )
  
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    out_mesh = trimesh.Trimesh(vertices, faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    out_mesh.apply_transform(rot)

    # Create the camera object
    H, W, _ = img.shape

    baseColorFactor = (1.0, 1.0, 0.95, 1.0)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=baseColorFactor)
    mesh = pyrender.Mesh.from_trimesh(
        out_mesh,
        material=material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_center = np.array([cx, cy])
    camera_transl = T.flatten()
    # Equivalent to 180 degrees around the y-axis. Transforms the fit to
    # OpenGL compatible coordinate system.
    camera_transl[0] *= -1.0

    camera_pose = np.eye(4)
    # camera_pose[:3, :3] = R
    camera_pose[:3, 3] = camera_transl

    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx, fy=fy,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)

    # Get the lights from the viewer
    # light_nodes = monitor.mv.viewer._create_raymond_lights()
    # for node in light_nodes:
    #     scene.add_node(node)

    r = pyrender.OffscreenRenderer(viewport_width=W,
                                   viewport_height=H,
                                   point_size=1.0)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]

    output_img = 255 - ((255 - color[:, :, :-1]) * valid_mask +
                        (1 - valid_mask) * img)

    output_img = (output_img * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, f'Frame_{frame_id}.jpg'), output_img)




# WARNING: Bugged
# This is only implemented for camera 01 with open3d.visualization.rendering
def project_smpl(frame_id, mesh):
    output_dir = prepare_out_dirs(dataDirs=['open3d_render_on_img'])[0]
    background_img = cv2.imread(os.path.join(DATA_DIR, 'cn01', '0000002150_color.jpg'), -1)

    H, W, _ = background_img.shape

    # create renderer
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    # set background to black
    render.scene.set_background([0.0, 0.0, 0.0, 1.0])


    # Define a simple unlit Material.
    # (The base color does not replace the arrows' own colors.)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # adds our mesh
    render.scene.add_geometry("mesh", mesh, mtl)

    # camera intrinsics
    camMat = np.array([
                    [1.02595850e+03, 0., 1.02028540e+03],
                    [0., 1.02400476e+03, 7.76271240e+02], 
                    [0., 0., 1. ]])

    # sets projection as in: https://stackoverflow.com/questions/70273002/project-3d-mesh-on-2d-image-using-camera-intrinsic-matrix
    near_plane = 0.01
    far_plane = 10.0
    render.scene.camera.set_projection(camMat, near_plane, far_plane, W, H)

    # Rotation matrix
    R = np.array([
        [-0.66923718,  0.74202016, -0.03908604],
        [ 0.50586555,  0.4164543 , -0.75542432],
        [-0.54426251, -0.52533031, -0.65406911]]
    )

    # Translation matrix
    T = np.array([
        [-0.22217233],
        [-0.29406874],
        [3.90767329]])
    
    # Position of our camera
    # Taken from: https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
    C = - R.T @ T

    center = [0, 0, 0]  # look_at target
    eye = C.flatten()  # camera position
    up = [0, 0, 1]  # camera orientation
    render.scene.camera.look_at(center, eye, up)



    # Read the image into a variable
    img_o3d = render.render_to_image()

    # convert image to opencv
    foreground_img = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)

    # making background transparent. taken from: 
    # https://stackoverflow.com/questions/40527769/removing-black-background-and-make-transparent-from-grabcut-output-in-python-ope
    tmp = cv2.cvtColor(foreground_img, cv2.COLOR_BGRA2GRAY)
    _, alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r, _ = cv2.split(foreground_img)
    rgba = [b,g,r, alpha]
    foreground_img = cv2.merge(rgba,4)
    # cv2.imwrite("test.jpg", dst)

    # merges two pictures, taken from: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    y1, y2 = 0, foreground_img.shape[0]
    x1, x2 = 0, foreground_img.shape[1]

    alpha_s = foreground_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        background_img[y1:y2, x1:x2, c] = (alpha_s * foreground_img[:, :, c] +
                                alpha_l * background_img[y1:y2, x1:x2, c])

    cv2.imwrite(os.path.join(output_dir, f'Frame_{frame_id}.png'), background_img)



def main():
    frame_id = 2150
    np.set_printoptions(suppress=True)
    # 3D - key points detected by voxel pose
    with open('data/pred_voxelpose.pkl', 'rb') as f:
        preds = pickle.load(f)

    vis = None

    # 3d keypoints predicted with voxelpose of this frame
    points_ = preds[frame_id]

    # this will also visualize smpl model in 3d space
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(points_, vis, frame_id)

    # renders smpl on 2d image and 3d space
    # if vis != None then it will produce smpl in 3d point cloud
    render_smpl(frame_id, vis)

    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    main()
