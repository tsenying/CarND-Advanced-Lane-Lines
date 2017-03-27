# import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import pickle
# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip

import config

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

lane_fits = {
    'left_fit': None,
    'right_fit': None
}

# find lane and plot lane on image
def process_image(image):
    lane_fits['left_fit'], lane_fits['right_fit'], binary_warped, lane_radius = find_lane( image, 
        CALIBRATION["mtx"], CALIBRATION["dist"], PERSPECTIVE["M"],
        left_fit = lane_fits['left_fit'], right_fit = lane_fits['right_fit'] )
    #image_with_lane = plot_lane( image, binary_warped, left_fit, right_fit, Minv, mtx, dist)
    image_with_lane = plot_lane( image, binary_warped, lane_fits['left_fit'], lane_fits['right_fit'], 
        PERSPECTIVE["Minv"], CALIBRATION["mtx"], CALIBRATION["dist"],
        lane_radius, REAL2PIXELS['xm_per_pix'])
    return image_with_lane

cap = cv2.VideoCapture('project_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

#shape=(720, 1280, 3)
width = 1280
height = 720

# M','J','P','G') -> .avi
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#lane_video = cv2.VideoWriter('lane_video.avi',fourcc,20.0,(width,height))

# http://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
# *’mp4v’ -> .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
lane_video = cv2.VideoWriter('lane_video.mp4',fourcc,fps,(width,height))

config.count = 0
start_frame = 380
stop_frame  = 400
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    config.count += 1
    if config.count%100 == 0:
        print("config.count={}".format(config.count))
        
    #if 250 <= config.count <= 378:
    if start_frame <= config.count <= stop_frame:
        if config.count%50 == 0:
            print("processing frame {}, shape={}".format(config.count, frame.shape))
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('./debug_images/orig/frame' + str(config.count) + '.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) )
        image_with_lane = process_image(frame_rgb)
        #cv2.imwrite('./debug_images/lane/frame' + str(config.count) + '.jpg', cv2.cvtColor(image_with_lane, cv2.COLOR_RGB2BGR) )

        lane_video.write( cv2.cvtColor(image_with_lane, cv2.COLOR_RGB2BGR) )
        
    if config.count > stop_frame:
        break

cap.release()
cv2.destroyAllWindows()
lane_video.release()
print("total config.count={}".format(config.count))
