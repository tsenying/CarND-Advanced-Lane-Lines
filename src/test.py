import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from lane_fit_utils import slidingWindowsPolyFit, lookAheadFilter, find_lane, plot_lane

# NOTE jpg and png format pixel values are different ranges? jpg = 8bit?
# image = mpimg.imread('./test_images/straight_lines1.jpg')
image = mpimg.imread('./test_images/test4.jpg')

plt.imshow(image)
plt.show()

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
CALIBRATION = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
PERSPECTIVE = pickle.load( open( "./camera_cal/perspective_transform_pickle.p", "rb"))

# Define conversions in x and y from pixels space to meters
REAL2PIXELS = {
  'ym_per_pix': 30/720, # meters per pixel in y dimension
  'xm_per_pix': 3.7/700 # meters per pixel in x dimension
}



# test find_lane with sliding windows
left_fit, right_fit, binary_warped, lane_radius = find_lane( image,
    CALIBRATION["mtx"], CALIBRATION["dist"], PERSPECTIVE["M"],
    )
    #plot=True )
print("left_fit={}".format(left_fit))
print("right_fit={}".format(right_fit))

plt.show()

# plot the lane on image
image_with_lane = plot_lane( image, binary_warped, left_fit, right_fit,
    PERSPECTIVE["Minv"], CALIBRATION["mtx"], CALIBRATION["dist"],
    lane_radius, REAL2PIXELS['xm_per_pix'])
plt.imshow(image_with_lane)
plt.show()

# test find_lane with look ahead filter
left_fit, right_fit, binary_warped, lane_radius = find_lane( image, 
    CALIBRATION["mtx"], CALIBRATION["dist"],  PERSPECTIVE["M"],
    left_fit=left_fit, right_fit=right_fit )
print("left_fit={}".format(left_fit))
print("right_fit={}".format(right_fit))

# plot the lane on image
image_with_lane = plot_lane( image, binary_warped, left_fit, right_fit, 
    PERSPECTIVE["Minv"], CALIBRATION["mtx"], CALIBRATION["dist"],
    lane_radius, REAL2PIXELS['xm_per_pix'])
plt.imshow(image_with_lane)
plt.show()
