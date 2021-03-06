import numpy as np
import cv2

import config

# Curvature of lane lines
import numpy as np
def lane_curvature( leftx, lefty, rightx, righty, y_eval ):
    """
    Find Lane Curvature
    
    Args:
        leftx (ndarray): left line x pixel positions
        lefty (ndarray): left line y pixel positions
        rightx
        righty
        y_eval (int): y pixel position to evaluate curvature (usually at bottom of image (binary_warped.shape[0] - 1), closest to vehicle)
        
    Returns:
        (left_radius, right_radius)
    """

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in real world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Calculate radius weighted by number of non-zero points on left and right lanes
    left_nonzeros = leftx.shape[0]
    right_nonzeros = rightx.shape[0]

    weighted_curverad = ((left_curverad * left_nonzeros) + (right_curverad * right_nonzeros))/(left_nonzeros + right_nonzeros)
    
    return (left_curverad, right_curverad, weighted_curverad)

def line_curvature( line_poly ):
    """
    Find line curvature
    
    Args:
        line_poly (array): line polynomial fit
    Returns:
        radius (float): in meters
    """
    line_poly1d_fn = np.poly1d( line_poly ) # turn coefficients into function
    
    # calculate set of points for poly to re-fit in real world
    # remember y axis is variant, x axis is function value
    y_points = np.array(np.linspace(0, config.image_shape['height'], num=50))
    x_points = np.array([ line_poly1d_fn(x) for x in y_points ])
    
    # evaluate curvature closest to vehicle
    y_eval = config.image_shape['height'] - 1
    
    # real world poly fit
    real_line_poly = np.polyfit(y_points * config.REAL2PIXELS['ym_per_pix'], x_points * config.REAL2PIXELS['xm_per_pix'], 2 )
    
    radius = ((1 + (2*real_line_poly[0]*y_eval*config.REAL2PIXELS['ym_per_pix'] + real_line_poly[1])**2)**1.5) / np.absolute(2*real_line_poly[0])

    return radius
    
# find off center distance and left and right line separation (lane width)
def lane_offset( left_fit, right_fit, warped_shape, xm_per_pix ):
    """
    Find Off Center Distance and Lane Width

    Args:
        left_fit, 
        right_fit,
        warped_shape (y_dim, x_dim): shape of warped image used for lane fit
        y_eval (int): y pixel position to evaluate curvature (usually at bottom of image (binary_warped.shape[0] - 1), closest to vehicle)
        xm_per_pix (int): meters per x pixel

    Returns:
        (off_center, lane_width)
    """

    y_dim, x_dim = warped_shape

    # # Calculate left and right x at bottom of image
    # # Approach 1. adjust pixel value result be xm_per_pix
    y_eval = y_dim - 1

    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_width = (right_x - left_x) * xm_per_pix

    assert 2 < lane_width < 5.5, "lane width should be reasonable: %r" % lane_width

    # off center distance
    lane_center = (right_x + left_x)/2

    off_center = (lane_center - (x_dim/2) ) * xm_per_pix

    half_image_width = ((x_dim/2) * xm_per_pix)
    assert 0.0 < abs(off_center) < half_image_width

    return off_center, lane_width



def slidingWindowsPolyFit( binary_warped ):
    """Sliding Windows Polynomial fit for lane lines
    
    Args:
        binary_warped (array): warped birds-eye view of lane image
    Returns:
        left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius
            - left_fit, right_fit: left and right lines polynomial fit
            - out_img: warped image with detected line pixels marked
            - left_radius, right_radius: radius of left and right lines
            - weighted_radius: combined left and right radius
    """
    
    # get histogram of the bottom half of the image 
    # - peaks used to get starting x-coord for left and right lines
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    # http://stackoverflow.com/questions/24739769/matplotlib-imshow-plots-different-if-using-colormap-or-rgb-array
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype(np.uint8)

    # Find peaks of left and right halves of histogram
    # These are used as the starting x-coord for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of sliding windows
    nwindows = 9

    # calculate height of each window
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
        
        # If found > minpix pixels, recenter next window on the mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

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
    
    left_radius, right_radius, weighted_radius = lane_curvature( leftx, lefty, rightx, righty, binary_warped.shape[0] - 1 )

    return left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius


# Assume you now have a new warped binary image
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
def lookAheadFilter( left_fit, right_fit, binary_warped, lane_image=False ):
    """Look Ahead Filter
        Given a fit from prior frame, use the fit to predicate the general region where current frame
        lane lines should be

    Args:
        left_fit (array): 2nd order polynomial fit of left lane line from previous frame
        right_fit (array): 2nd order polynomial fit of right lane line from previous frame
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

    # Extract left and right line pixel positions
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
        ploty = np.linspace(20, binary_warped.shape[0]-1, binary_warped.shape[0]-20 ) # start at 20 to avoid x out of bounds
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        ploty_int = ploty.astype(np.uint16)
        left_fitx_int = left_fitx.astype(np.uint16)
        right_fitx_int = right_fitx.astype(np.uint16)
        
        if (left_fitx_int.max() > 1279):
            config.debug_log.write("lookAheadFilter count={}, left_fit={}\n".format(config.count, left_fit))
            config.debug_log.write("lookAheadFilter count={}, ploty.shape={}, left_fitx_int.max={}, right_fitx_int.max={}\n".format(config.count, ploty.shape, left_fitx_int.max(), right_fitx_int.max()))
        else:
            out_img[ ploty_int, left_fitx_int ] = [255,255,0]
            
        if (right_fitx_int.max() > 1279):
            config.debug_log.write("lookAheadFilter count={}, right_fit={}\n".format(config.count, right_fit))
            config.debug_log.write("lookAheadFilter count={}, ploty.shape={}, left_fitx_int.max={}, right_fitx_int.max={}\n".format(config.count, ploty.shape, left_fitx_int.max(), right_fitx_int.max()))
        else:
            out_img[ ploty_int, right_fitx_int ] = [255,255,0]

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

    # get the left and right line radius
    left_radius, right_radius, weighted_radius = lane_curvature( leftx, lefty, rightx, righty, binary_warped.shape[0] - 1 )
    
    return left_fit, right_fit, out_img, left_radius, right_radius, weighted_radius

def are_lines_parallel( line_one_poly, line_two_poly, threshold=(0.0005, 0.55) ):
    """
    check if two lines are parallel by comparing first two polynomial fit coefficients
    - coefficients should be within margins
    
    Args:
        param other_line (array): line to compare polynomial coefficients
        threshold (tuple): floats representing delta thresholds for coefficients
    
    Returns:
        boolean
    """
    diff_first_coefficient = np.abs(line_one_poly[0] - line_two_poly[0])
    diff_second_coefficient = np.abs(line_one_poly[1] - line_two_poly[1])

    is_parallel = diff_first_coefficient < threshold[0] and diff_second_coefficient < threshold[1]

    return is_parallel

def line_distance(line_one_poly, line_two_poly, y_eval):
    """
    get distance between current fit with other_line
    
    Args:
        line_one_poly: line one polynomial 
        line_two_poly:
        y_eval: value to evaluate
    Returns:
        float
    """
    line_one_poly1d = np.poly1d( line_one_poly )
    line_two_poly1d = np.poly1d( line_two_poly )
    return np.abs( line_one_poly1d( y_eval ) - line_two_poly1d( y_eval ))


