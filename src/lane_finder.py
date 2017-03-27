from lane import Lane
from line import Line

from lane_fit_utils import slidingWindowsPolyFit, lookAheadFilter, plot_lane
from combined_binary_util import colorAndGradientThresholdBinary
from image_utils import image_warp
import matplotlib.pyplot as plt

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

