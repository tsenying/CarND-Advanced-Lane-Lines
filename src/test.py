import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from lane_fit_utils import slidingWindowsPolyFit, lookAheadFilter, find_lane

# NOTE jpg and png format pixel values are different ranges? jpg = 8bit?
image = mpimg.imread('./test_images/straight_lines1.jpg')

plt.imshow(image)
plt.show()

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
CALIBRATION = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
PERSPECTIVE = pickle.load( open( "./camera_cal/perspective_transform_pickle.p", "rb"))

left_fit, right_fit, binary_warped = find_lane( image,
    CALIBRATION["mtx"], CALIBRATION["dist"], PERSPECTIVE["M"],
    plot=True )
print("left_fit={}".format(left_fit))
print("right_fit={}".format(right_fit))
