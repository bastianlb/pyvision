import cv2
import numpy as np
from pathlib import Path
import json
from utils_ksv1 import load_camera_params, load_cam_infos
from utils_ksv1 import project_pose, homogenous_to_rot_trans, project_to_2d

CAMERAS = ["camera01", "camera02", "camera03", "camera04"]

def undistort_images(image_folder: Path, output_folder: Path, camera_parameters: dict):
    """
    Undistort images in a folder based on camera parameters.
    
    Args:
        image_folder (Path): Path to the folder containing images.
        output_folder (Path): Path to the folder to save undistorted images.
        camera_parameters (dict): Dictionary containing camera intrinsics and distortion coefficients.
    """
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cam_params = camera_parameters
    
    # Extract required parameters
    intrinsics = cam_params['intrinsics']
    width, height = cam_params['width'], cam_params['height']
    radial = cam_params['radial_params']
    tangential = cam_params['tangential_params']
    
    # Combine radial and tangential distortion parameters
    distortion_coeffs = np.array([*radial, *tangential], dtype=np.float32)
    
    # Iterate through all images in the folder
    for image_path in sorted(image_folder.glob('*.tiff')):  # Change '*.jpg' to match your file type
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue
        
        # Get optimal new camera matrix
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion_coeffs, (width, height), alpha=0, newImgSize=(width, height)
        )

        print(new_camera_matrix)
        
        # Undistort the image
        undistorted_image = cv2.undistort(image, intrinsics, distortion_coeffs, None, new_camera_matrix)
        
        # Save the undistorted image
        output_path = output_folder / image_path.name
        cv2.imwrite(str(output_path), undistorted_image)
        print(f"Saved undistorted image to {output_path}")

# Example usage:
if __name__ == "__main__":
    # Define paths
    images_folder = Path("/home/victorkawai/121224_fornero_take_6/ksv1capture/export/depth/")  # Folder with images
    output_folder = Path("/home/victorkawai/121224_fornero_take_6/ksv1capture/export/depth_undistorted/")  # Folder for undistorted images
    take_path = Path("/home/victorkawai/121224_fornero_take_6/ksv1capture/export/")  # Folder containing calibration data
    
    for cam in CAMERAS[:]:
    # Load camera parameters
        camera_params = load_cam_infos(take_path)[cam]
        
        # Undistort images
        undistort_images(images_folder, output_folder, camera_params)
