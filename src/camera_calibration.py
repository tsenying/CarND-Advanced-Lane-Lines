# ref: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

######## Camera Calibration ########

### 1. Find chessboard corners

# calibration images has 9x6 inner corners
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(nx,ny,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)


cv2.destroyAllWindows()

#http://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
for i in range (1,5):
    cv2.waitKey(1)


### 2. Do camera calibration given object points and image points
import pickle

img_1 = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img_1.shape[1], img_1.shape[0]) # dimensions reversed
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "./camera_cal/calibration_pickle.p", "wb" ) )


### Test undistortion on an image
img = cv2.imread('./camera_cal/calibration1.jpg')

# Visualize undistortion
img_undistort = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('./output_images/calibration_undistort.jpg', img_undistort)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(img_undistort)
ax2.set_title('Undistorted Image', fontsize=20)

f.savefig('./output_images/camera_calibration.jpg')
plt.show()

### Try calibration on a road image
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
CALIBRATION = {'mtx': mtx, 'dist': dist}

# Read in an image
img_orig = mpimg.imread('test_images/straight_lines1.jpg')

# 1) Undistort using mtx and dist
img = cv2.undistort(img_orig, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.tight_layout()
ax1.imshow(img_orig)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(img)
ax2.set_title('Undistorted', fontsize=20)

f.savefig('./output_images/straight_lines1_undistort.jpg')
plt.show()
