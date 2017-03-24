import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from lane_fit_utils import slidingWindowsPolyFit, lookAheadFilter, find_lane, plot_lane

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
CALIBRATION = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
PERSPECTIVE = pickle.load( open( "./camera_cal/perspective_transform_pickle.p", "rb"))
# Define conversions in x and y from pixels space to meters
REAL2PIXELS = {
  'ym_per_pix': 30/720, # meters per pixel in y dimension
  'xm_per_pix': 3.7/700 # meters per pixel in x dimension
}
    
# find lane and plot lane on image
def process_image(image):
    left_fit, right_fit, binary_warped, lane_radius = find_lane( image, 
        CALIBRATION["mtx"], CALIBRATION["dist"], PERSPECTIVE["M"] )
    #image_with_lane = plot_lane( image, binary_warped, left_fit, right_fit, Minv, mtx, dist)
    image_with_lane = plot_lane( image, binary_warped, left_fit, right_fit, 
        PERSPECTIVE["Minv"], CALIBRATION["mtx"], CALIBRATION["dist"],
        lane_radius, REAL2PIXELS['xm_per_pix'])
    return image_with_lane

project_output = 'project_output_video.mp4'
clip = VideoFileClip("project_video.mp4") #.subclip(0,15)
project_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
project_clip.write_videofile(project_output, audio=False)
