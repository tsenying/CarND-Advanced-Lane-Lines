import numpy as np
import cv2
from sobel_utils import abs_sobel_thresh, mag_thresh, dir_threshold

def colorAndGradientThresholdBinary(img, color_thresh=(170, 255), sobel_thresh=(20, 100), ksize=3):
    """Use Sobel gradient and Color transforms to create a thresholded binary image

    Args:
        img (numpy.ndarray): Source image. Color channels in RGB order.
        color_thresh (tuple): Color channel threshold. Two element tuple.
        sobel_thresh (tuple): Sobel operator threshold. Two element tuple.
        ksize (int): Sobel operator kernel size.

    Returns:
        (color_binary, sx_binary, s_binary): color_binary, sx_binary, s_binary
    """
    img = np.copy(img)

    ### Sobel Gradient in x direction ( x direction accentuates vertical lines )
    #sx_binary = abs_sobel_thresh(img, orient='x', ksize=3, thresh_min=sobel_thresh[0], thresh_max=sobel_thresh[1])

    ### Sobel magnitude
    mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=sobel_thresh) # (30,100) seems to work well

    ### Sobel direction (0.5 to 1.4) is about (30 to 80 degrees)
    dir_binary = dir_threshold(img, sobel_kernel=13, thresh=(0.5, 1.5))

    ### Combine magnitude and direction where both thresholds are satisfied
    # (note that magnitude and direction takes both individual Sobel x and y into account)
    sobel_combined_binary = np.zeros_like(dir_binary)
    sobel_combined_binary[((mag_binary == 1) & (dir_binary == 1))] = 1

    ### Color Channel
    # Convert to HLS color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    s_gradiented_binary = ((s_binary == 1) & (dir_binary == 1)) # s_binary filtered by gradient

    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= 15) & (h_channel <= 30)] = 1

    color_combined_binary = np.zeros_like(s_binary)
    color_combined_binary[ (s_binary == 1) & ( h_binary == 1)] =1

    color_gradiented_binary = ((color_combined_binary == 1) & (dir_binary == 1)) # s_binary filtered by gradient

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sobel_combined_binary), sobel_combined_binary, color_gradiented_binary))

    # Combine sobel and color results
    combined_binary = np.zeros_like(sobel_combined_binary)
    combined_binary[ (sobel_combined_binary == 1) | (color_gradiented_binary == 1) ] = 1

    return combined_binary, color_binary, \
        sobel_combined_binary, mag_binary, dir_binary, \
        s_gradiented_binary, s_binary, h_binary, color_gradiented_binary

# plot lane overlay on original image
def plot_lane( image, binary_warped, left_fit, right_fit, Minv, mtx, dist):
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
    return image_with_lane
