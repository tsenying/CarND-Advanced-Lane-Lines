
import cv2

# import matplotlib.image as mpimg
import pickle
# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip

import config
from lane_finder import LaneFinder

# Read in the saved camera matrix and distortion coefficients
# These are the arrays calculated using cv2.calibrateCamera()
CALIBRATION = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
PERSPECTIVE = pickle.load( open( "./camera_cal/perspective_transform_pickle.p", "rb"))

# Define conversions in x and y from pixels space to meters
REAL2PIXELS = {
  'ym_per_pix': 30/720, # meters per pixel in y dimension
  'xm_per_pix': 3.7/700 # meters per pixel in x dimension
}

config.debug_log = open('debug.log', 'w')

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

config.lane_finder = LaneFinder( CALIBRATION, PERSPECTIVE, REAL2PIXELS )
config.count = 0
start_frame = 500 #380
stop_frame  = 550 #400
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
        image_with_lane = config.lane_finder.process_image(frame_rgb)
        #cv2.imwrite('./debug_images/lane/frame' + str(config.count) + '.jpg', cv2.cvtColor(image_with_lane, cv2.COLOR_RGB2BGR) )

        lane_video.write( cv2.cvtColor(image_with_lane, cv2.COLOR_RGB2BGR) )
        
    if config.count > stop_frame:
        break

cap.release()
cv2.destroyAllWindows()
lane_video.release()

if config.debug_log is not None:
    config.debug_log.close()
    
print("total config.count={}".format(config.count))
