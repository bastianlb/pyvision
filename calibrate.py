import sys
from pathlib import Path

import numpy as np
import cv2
from cv2 import aruco

print(cv2.__version__)

CAMERAS = ["camera01", "camera02", "camera03", "camera04"]

if __name__ == "__main__":
    dir = Path("/home/victorkawai/121224_fornero_take_6/ksv1capture/export/")
    if not dir.exists():
        print("Invalid directory")
        sys.exit(1)

    cv2.namedWindow('out', cv2.WINDOW_GUI_NORMAL)

    images = []
    for cam in CAMERAS:
        frame = 5
        file_id = str(frame).zfill(6)
        fname = f"color_{file_id}_{cam}.jpg"
        image = dir / "color" / fname
        print(image)
        if not image.exists():
            print("File not found")
            sys.exit(1)

        array = cv2.imread(str(image))
        if array is None:
            print(f"Could not interpret {image} as an image")
        images.append(array)

        params = aruco.DetectorParameters()
        # params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
        rows = 4
        cols = 3
        square_length = 0.088  # Size of each square (in meters)
        marker_length = 0.0655  # Size of each marker (in meters)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        board = aruco.CharucoBoard((rows, cols), square_length, marker_length, dictionary)

        aruco_corners, aruco_ids, aruco_points = cv2.aruco.detectMarkers(array, dictionary, None, None, params)

        if len(aruco_corners) == 0:
            print("No corners detected")
            for marker in aruco_points:
                for (x, y) in np.int32(marker.reshape(marker.shape[1:])):
                    cv2.circle(array, (x, y), 10, (0, 255, 0), 1)
        else:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                aruco_corners,
                aruco_ids,
                array,
                board)
            aruco.drawDetectedCornersCharuco(array, charuco_corners, charuco_ids, (255, 0, 0))

        cv2.imshow("out", array)
        cv2.imwrite(cam + "charuco_detection.jpg", array)
        cv2.waitKey(0)

    cv2.destroyAllWindows()




