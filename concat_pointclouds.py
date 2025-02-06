import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

print(o3d.__version__)

DATA_DIR = "/home/victorkawai/121224_fornero_take_6/ksv1capture/export/pointclouds_e57/"
CAMERAS = ["camera01", "camera02", "camera03", "camera04"]


def render_camera_poses(vis):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry("coordinate_frame", mesh_frame)
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(6)
        fpath = os.path.join(DATA_DIR, f"pointcloud_{file_id}_{cam}.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        vis.add_geometry(f"{cam}-ply", ply)


if __name__ == "__main__":
    frame_id = 5
    np.set_printoptions(suppress=True)

    # new extrinsics
    # point = np.array([0.249721, -0.005661, -0.974014])
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    render_camera_poses(vis)
    app.add_window(vis)
    app.run()
