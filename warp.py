import os
import numpy as np
import cv2


def warp_birds_eye(image, region):
    IMAGE_H, IMAGE_W, _ = image.shape
    cropped = image[region[0][1]:region[1][1], region[0][0]:region[1][0]]
    cv2.imshow("image", cropped)
    cv2.waitKey(0)
    src = np.float32([
        [region[0][0], region[0][1]],
        [region[1][0], region[0][1]],
        [region[1][0], region[1][1]],
        [region[0][0], region[1][1]],
    ])
    disp = 320
    dst = np.float32([
        [0, 0],
        [IMAGE_W, 0],
        [IMAGE_W - disp, IMAGE_H],
        [0 + disp, IMAGE_H - 1]
    ])

    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
    # Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

    warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))

    return warped_img


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))


if __name__ == "__main__":
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)
    dataset = '/data/carla/_out_Town01'
    frame = 100
    fname = os.path.join(dataset, f'{frame:08}_rgb.png')
    image = cv2.imread(fname)  # Read the test img
    if image is None:
        print(f"Image {fname} not found")
        exit(1)
    top_view_name = os.path.join(dataset, f'{frame:08}_rgb_top.png')
    topview = cv2.imread(top_view_name)  # Read the test img
    if topview is None:
        print(f"Image {top_view_name} not found")
        exit(1)
    clone = image.copy()
    refPt = []
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt = []
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            if (len(refPt) == 2):
                break
            else:
                refPt = []
                print("Try again..")
        elif key == ord("e"):
            exit(1)
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    def putText(image, text):
        cv2.putText(image, text, (int(image.shape[1] / 2) - 50, 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (218, 105, 0), 2)

    # close all open windows
    warped = warp_birds_eye(clone.copy(), refPt)
    putText(topview, "Topview..")
    putText(warped, "Warped..")
    vis = np.concatenate((topview, warped), axis=1)
    cv2.imshow("warped", vis)
    cv2.imwrite(fname.split(".")[0] + "_combined.jpg", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
