# import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import pickle
# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip

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

cap = cv2.VideoCapture('project_video.mp4')

#shape=(720, 1280, 3)
width = 1280
height = 720

# M','J','P','G') -> .avi
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#lane_video = cv2.VideoWriter('lane_video.avi',fourcc,20.0,(width,height))

# *’mp4v’ -> .*’mp4v’
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
lane_video = cv2.VideoWriter('lane_video.mp4',fourcc,20.0,(width,height))

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count%100 == 0:
        print("count={}".format(count))
        
    #if 250 <= count <= 378:
    if 250 <= count <= 300:
        print("processing frame {}, shape={}".format(count, frame.shape))
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./debug_images/orig/frame' + str(count) + '.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) )
        image_with_lane = process_image(frame_rgb)
        cv2.imwrite('./debug_images/lane/frame' + str(count) + '.jpg', cv2.cvtColor(image_with_lane, cv2.COLOR_RGB2BGR) )

        lane_video.write( cv2.cvtColor(image_with_lane, cv2.COLOR_RGB2BGR) )
        
    if count > 378:
        break

cap.release()
cv2.destroyAllWindows()
lane_video.release()
print("total count={}".format(count))
