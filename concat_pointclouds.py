import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

DATA_DIR = "/data/develop/export_mkv_k4a/pointcloud_export"
CAMERAS = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]


def render_camera_poses(vis):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry("coordinate_frame", mesh_frame)
    for cam in CAMERAS:
        file_id = str(frame_id).zfill(4)
        fpath = os.path.join(DATA_DIR, cam, f"{file_id}_pointcloud.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        vis.add_geometry(f"{cam}-ply", ply)


if __name__ == "__main__":
    frame_id = 1100
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
