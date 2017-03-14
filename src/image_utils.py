import cv2
import numpy as np

def image_warp(img, mtx, dist, M):
    """Warp image using perspective transform

    First, undistort image using camera calibration

    Args:
        img (numpy.ndarray): Source image. Color channels in RGB order.
        mtx: camera matrix,
        dist: distortion coefficients
        M: the transform matrix

    Returns:
        (warped): warped image
    """

    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img_size = (img.shape[1], img.shape[0]) # note switch of x,y order

    # 3) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
