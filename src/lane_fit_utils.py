import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_utils import image_warp

from combined_binary_util import colorAndGradientThresholdBinary

def slidingWindowsPolyFit( binary_warped ):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    # http://stackoverflow.com/questions/24739769/matplotlib-imshow-plots-different-if-using-colormap-or-rgb-array
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            #print("window={}, leftx_current={}".format(window, leftx_current))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            #print("window={}, rightx_current={}".format(window, rightx_current))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # for output image, color left lane points as red, right lane points as blue
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    # note that the y,x params ordering int numpy.polyfit
    # "fitting for f(y), rather than f(x),
    # because the lane lines in the warped image are near vertical and
    # may have the same x value for more than one y value
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ###print("lefty.shape={}, left_fit={}".format(lefty.shape, left_fit))
    ###print("righty.shape={}, right_fit={}".format(righty.shape, right_fit))

    return left_fit, right_fit, out_img


# Assume you now have a new warped binary image
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
# (use the same image just to try out)
def lookAheadFilter( left_fit, right_fit, binary_warped, lane_image=False ):
    """Look Ahead Filter

    Args:
        left_fit (array): 2nd order polynomial fit of left lane line from previous frame
        right_fit (array):
        binary_warped (numpy.ndarray): thresholded binary
        lane_image (boolean): create lane image

    Return:
        left_fit: 2nd order polynomial fit of left lane line for current frame
        right_fit:
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    # Get non-zeros within margin to either side of previous frame left, right lines
    left_line = left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]
    left_lane_inds = ((nonzerox > (left_line - margin)) & (nonzerox < (left_line + margin)))

    right_line = right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]
    right_lane_inds = ((nonzerox > (right_line - margin)) & (nonzerox < (right_line + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if lane_image:
        # Create an image to draw on and an image to show the selection window
        out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[ ploty.astype(np.uint16), left_fitx.astype(np.uint16) ] = [255,255,0]
        out_img[ ploty.astype(np.uint16), right_fitx.astype(np.uint16) ] = [255,255,0]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    else:
        out_img = None

    return left_fit, right_fit, out_img

## Collect transforms into pipeline function
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
def find_lane( image, mtx, dist, M, left_fit=None, right_fit=None, plot=False ):
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
        left_fit, right_fit, out_img = slidingWindowsPolyFit( binary_warped )
        fit_title = 'Sliding window fit'
        ###print("Sliding window fit left_fit={}, right_fit={}".format(left_fit, right_fit))
    else:
        left_fit, right_fit, out_img = lookAheadFilter( left_fit, right_fit, binary_warped, lane_image=plot )
        fit_title = 'Look ahead filter fit'
        ###print("Look ahead filter fit left_fit={}, right_fit={}".format(left_fit, right_fit))

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

    return left_fit, right_fit, binary_warped