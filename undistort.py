import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import os

# Step 1: Load the JSON file
# json_file = '/home/victorkawai/121224_fornero_take_6/ksv1capture/export/calibration/camera01.json'
# with open(json_file, 'r') as file:
#     data = json.load(file)

def precompute_undistort_maps(json_params, width, height):
    # Camera matrix
    camera_matrix = np.array([
        [json_params['fov_x'], 0, json_params['c_x']],
        [0, json_params['fov_y'], json_params['c_y']],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.array([
        json_params['radial_distortion']['m00'],  # k1
        json_params['radial_distortion']['m10'],  # k2
        json_params['tangential_distortion']['m00'],  # p1
        json_params['tangential_distortion']['m10'],  # p2
        json_params['radial_distortion']['m20'],  # k3
        json_params['radial_distortion']['m30'],  # k4 (optional)
        json_params['radial_distortion']['m40'],  # k5 (optional)
        json_params['radial_distortion']['m50'],  # k6 (optional)
    ], dtype=np.float32)

    # Create new camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 0)

    # Precompute maps
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (width, height), cv2.CV_32F
    )

    json_params['new_camera_matrix'] = new_camera_matrix.tolist()
    json_params['original_camera_matrix'] = camera_matrix.tolist()
    json_params['distortion_coefficients'] = dist_coeffs.tolist()
    return map1, map2


# Step 2: Define a function to process camera parameters
def undistort_and_update(json_params, input_image_path, output_image_path, map1, map2):

    # # Load the input image
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

    # Apply the rectification maps to the image
    undistorted_image = cv2.remap(input_image, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Save the undistorted image
    cv2.imwrite(output_image_path, undistorted_image)

    print(f"Undistorted image saved as {output_image_path}")

cur_folder = os.path.dirname(os.path.abspath(__file__))

base_folder = os.path.join(cur_folder, "data")
# output_folder = '/home/victorkawai/121224_fornero_take_6/ksv1capture/export/'
calibration_folder = os.path.join(base_folder, "calibration")

for i in ['color', 'depth']:
    for j in ['camera01', 'camera02', 'camera03', 'camera04']:
        image_files = sorted(glob.glob(f'{base_folder}/{i}/*{j}*'))
        json_file = f'{calibration_folder}/{j}.json'

        for image_file in image_files:
            # print(image_file)
            output_image_path = image_file
            output_image_path = image_file.replace('color_dist', 'color').replace('depth_dist', 'depth')
            with open(json_file, 'r') as file:
                data = json.load(file)
            json_params = data['value0'][f'{i}_parameters']

            map1, map2 = precompute_undistort_maps(json_params, json_params['width'], json_params['height'])

            undistort_and_update(json_params, image_file, output_image_path, map1, map2)
            with open(json_file, 'w') as file:
                json.dump(data, file, indent=4)