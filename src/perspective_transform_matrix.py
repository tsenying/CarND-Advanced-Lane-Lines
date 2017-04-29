# Calculate the perspective transform matrix M and inverse Minv
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.path import Path
import matplotlib.patches as patches

from image_utils import image_warp

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = mpimg.imread('test_images/straight_lines1.jpg')

# Define 4 source points 
# top left [622,435]
# top right [662,435]
# bottom right [1040,675]
# bottom left [272,675]
src_points = [[596,450],[685,450],[1100,720],[200,720]]
src = np.float32(src_points)

# Define 4 destination points
dst_points = [
    [320, 0], 
    [960, 0], 
    [960, 720], 
    [320, 720]]
dst = np.float32(dst_points)

# Use cv2.getPerspectiveTransform() to get M, the transform matrix
M = cv2.getPerspectiveTransform(src, dst)

# Inverse transform matrix for transforming warped back to perspective view
Minv = cv2.getPerspectiveTransform(dst, src)

# Save the perspective transform result for later use 
pXform_pickle = {}
pXform_pickle["M"] = M
pXform_pickle["Minv"] = Minv
pickle.dump( pXform_pickle, open( "./camera_cal/perspective_transform_pickle.p", "wb" ) )

### Test out the perspective transform
# warp image
img_warped = image_warp(img, mtx, dist, M)

# display result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.tight_layout()
ax1.imshow(img)

# draw the src points boundary
verts = np.float32( src_points + [src_points[0]] )
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]

path = Path(verts, codes)
patch = patches.PathPatch(path, edgecolor='red', facecolor='none', lw=2)
ax1.add_patch(patch)

ax1.set_title('Original Image', fontsize=20)

ax2.imshow( img_warped )

# draw the dst points boundary
verts = np.float32( dst_points + [dst_points[0]] )

path = Path(verts, codes)
patch = patches.PathPatch(path, edgecolor='red', facecolor='none', lw=2)
ax2.add_patch(patch)

ax2.set_title('Undistorted and Warped Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

fig.savefig('./output_images/perspective_transform_test.jpg')
#plt.show()

cv2.imwrite('./output_images/straight_lines1.jpg', img)
cv2.imwrite('./output_images/straight_lines1_warped.jpg', img_warped)
