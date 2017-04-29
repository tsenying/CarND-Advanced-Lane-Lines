import numpy as np
import cv2
import matplotlib.pyplot as plt

from lane import Lane
from line import Line

from lane_fit_utils import slidingWindowsPolyFit, lookAheadFilter, lane_offset, line_curvature
from combined_binary_util import colorAndGradientThresholdBinary
from image_utils import image_warp

import config

class LaneFinder():
    """ Find lane in image
    """
    nframes = 5
    
    def __init__( self, CALIBRATION, PERSPECTIVE, REAL2PIXELS ):
        self.lane = Lane(self.nframes)
        self.CALIBRATION = CALIBRATION
        self.PERSPECTIVE = PERSPECTIVE
        self.REAL2PIXELS = REAL2PIXELS
        
    ## Collect transforms into pipeline function
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    def find_lane( self, image, mtx, dist, M, left_fit=None, right_fit=None ):
        """
        Find Lane

        Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

        If lane line polynomial fit is available from previous frame, for current frame poly fit
          Use look ahead filter method
        else:
          Use sliding window method

        Args:
            image (numpy.ndarray): Source image. Color channels in RGB order.
            mtx: camera matrix,
            dist: distortion coefficients
            M: the transform matrix
            left_fit: average polyfit from previous frames
            right_fit: average polyfit from previous frames

        Returns:
            (left_fit, right_fit, warped): warped image
        """

        # 1. color and gradient binary
        threshold_binaries = \
            colorAndGradientThresholdBinary(image, color_thresh=(140, 255), sobel_thresh=(50, 135), ksize=5)
        combined_binary = threshold_binaries[0]

        # 2. undistort and warp image
        binary_warped = image_warp(combined_binary, mtx, dist, M)

        # 3. Sliding windows fit on first image
        if left_fit is None:
            left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius = \
              slidingWindowsPolyFit( binary_warped )
        else:
            # Look ahead filter fit if line fit is available
            left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius = \
              lookAheadFilter( left_fit, right_fit, binary_warped, lane_image=True )

        return left_fit, right_fit, binary_warped, weighted_radius, out_img

    # plot lane overlay on original image
    def plot_lane( self, image, binary_warped, left_fit, right_fit, Minv, mtx, dist, lane_radius, xm_per_pix):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Project lines on original image

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
        image_with_lane = cv2.addWeighted(image_undistorted, 1, newwarp, 0.3, 0)

        # Add lane offset
        off_center, lane_width = lane_offset( left_fit, right_fit, binary_warped.shape, xm_per_pix )

        # curve to left or right
        if (left_fit[0] < 0):
            curve_dir = 'L'
        else:
            curve_dir = 'R'

        annotate_str = "Radius {0:.2f}({1}), Off-Center {2:.3f}".format(lane_radius, curve_dir, off_center)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_with_lane,annotate_str,(10,100), font, 2,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(image_with_lane,"Frame:{}".format(config.count), (10,150), font, 1, (255,255,255), 2 ,cv2.LINE_AA )

        return image_with_lane


    # find lane and plot lane on image
    def process_image(self, image):
        # use line.best_fit if available
        self.lane.left_line.current_fit, self.lane.right_line.current_fit, binary_warped, lane_radius, out_img = self.find_lane( image, 
            self.CALIBRATION["mtx"], self.CALIBRATION["dist"], self.PERSPECTIVE["M"],
            left_fit = self.lane.left_line.best_fit, right_fit = self.lane.right_line.best_fit )
            
        is_valid = self.lane.update()
        if not is_valid:
            config.debug_log.write("LaneFinder#process_image Frame {} invalid\n".format( config.count ))
            invalid_frame_image = self.plot_lane( image, binary_warped, self.lane.left_line.current_fit, self.lane.right_line.current_fit, 
                self.PERSPECTIVE["Minv"], self.CALIBRATION["mtx"], self.CALIBRATION["dist"],
                lane_radius, self.REAL2PIXELS['xm_per_pix'])
            cv2.imwrite('./debug_images/invalid/frame' + str(config.count) + '.jpg', cv2.cvtColor(invalid_frame_image, cv2.COLOR_RGB2BGR) )
            cv2.imwrite('./debug_images/invalid/lanes/frame' + str(config.count) + '.jpg', cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            
        # Calculate lane radius using best fit
        left_radius = line_curvature( self.lane.left_line.best_fit )
        right_radius = line_curvature( self.lane.right_line.best_fit )
        lane_radius = (left_radius + right_radius)/2
        
        image_with_lane = self.plot_lane( image, binary_warped, self.lane.left_line.best_fit, self.lane.right_line.best_fit, 
            self.PERSPECTIVE["Minv"], self.CALIBRATION["mtx"], self.CALIBRATION["dist"],
            lane_radius, self.REAL2PIXELS['xm_per_pix'])
        return image_with_lane

        

