import numpy as np
import cv2
import matplotlib.pyplot as plt

from lane import Lane
from line import Line

from lane_fit_utils import slidingWindowsPolyFit, lookAheadFilter, lane_offset
from combined_binary_util import colorAndGradientThresholdBinary
from image_utils import image_warp

import config

class LaneFinder():
    nframes = 8
    
    def __init__( self ):
        self.lane = Lane(self.nframes)
        
    ## Collect transforms into pipeline function
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    def find_lane( self, image, mtx, dist, M, left_fit=None, right_fit=None, plot=False ):
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
            left_fit: polyfit from previous frame
            right_fit:

        Returns:
            (left_fit, right_fit, warped): warped image
        """

        ###print("type(image)={}, image.shape={}".format( type(image), image.shape ))

        # 1. color and gradient binary
        threshold_binaries = \
            colorAndGradientThresholdBinary(image, color_thresh=(128, 255), sobel_thresh=(50, 135), ksize=5)
        combined_binary = threshold_binaries[0]

        if plot:
            # Plot the color and gradient result
            f, axes = plt.subplots(1, 2, squeeze=False, figsize=(24, 9))
            f.suptitle('Gradient and Color Threshold', fontsize=32)
            f.tight_layout()
            axes[0][0].imshow(image)
            axes[0][0].set_title('Original Image - undistorted', fontsize=40)
            axes[0][1].imshow(combined_binary, cmap='gray')
            axes[0][1].set_title('Combined Binary', fontsize=40)

        # 2. undistort and warp image
        binary_warped = image_warp(combined_binary, mtx, dist, M)

        if plot:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.suptitle('Undistort', fontsize=32)
            f.tight_layout()
            ax1.imshow(combined_binary, cmap='gray')
            ax1.set_title('Combined Binary', fontsize=40)
            ax2.imshow(binary_warped, cmap='gray')
            ax2.set_title('Undistorted and Warped', fontsize=40)

        # 3. Sliding windows fit on first image
        if left_fit is None:
            left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius = \
              slidingWindowsPolyFit( binary_warped )
            fit_title = 'Sliding window fit'
            # print("Sliding window fit left_fit={}, right_fit={}".format(left_fit, right_fit))
        else:
            # Look ahead filter fit if line fit is available
            left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius = \
              lookAheadFilter( left_fit, right_fit, binary_warped, lane_image=plot )
            fit_title = 'Look ahead filter fit'
            # print("Look ahead filter fit left_fit={}, right_fit={}".format(left_fit, right_fit))

        if plot:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

            # left and right lines from polyfit over y range
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # color left lane points as red, right lane points as blue
            #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            f, (ax1) = plt.subplots(1,1, figsize=(6,4))
            f.tight_layout()
            ax1.imshow(out_img)
            ax1.plot(left_fitx, ploty, color='yellow')
            ax1.plot(right_fitx, ploty, color='orange')
            ax1.set_title(fit_title, fontsize=40)

        return left_fit, right_fit, binary_warped, weighted_radius

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
        ###print("color_warp.shape={}".format(color_warp.shape))

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
        ###print("off_center={}, lane_width={}".format(off_center, lane_width))

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


