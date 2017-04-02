
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Project Files

- Project Report: Advanced_Lane_Finding.md (this file)
- Source Code: in `./src` directory
- Supporting Images: in `./output_images` directory
- Camera calibration data: `camera_cal/calibration_pickle.p`
- Perspective transformation data: `camera_cal/perspective_transform_pickle.p`
- Jupyter notebook used for initial investigations: `AdvancedLaneFinding.ipynb`
- Final video with lane overlay: `lane_video.mp4` (not in git repository, uploaded to https://youtu.be/zp9oSdVjiws)

### Running source files
1. generate camera calibration: `python src/camera_calibration.py`
2. generate perspective transformation matrices: `python src/perspective_transform_matrix.py`
3. process video file: `src/process_video_images.py`

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.jpg "Camera Calibration"
[image2]: ./output_images/straight_lines1_undistort.jpg "Road Image Undistorted"
[image3]: ./output_images/combined_sobel_and_color_binary.jpg "Combined Sobel and Color Binary Example"
[image4]: ./output_images/perspective_transform_test.jpg "Warp Example"
[image5]: ./output_images/sliding_window_fit.jpg "Sliding Window Fit"
[image6]: ./output_images/look_ahead_filter.jpg  "Look Ahead Filter Fit"
[image7]: ./output_images/frame755.jpg "Lane Overlay with annotation"
[video1]: ./lane_video.mp4 "Video"

## Camera Calibration

#### 1. Camera calibration is done using [OpenCV Camera Calibration support](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration)

Camera calibration is needed to correct for distortion of images by the camera.
What is needed is referred to by the reference as the 'distortion coefficients' and the 'camera matrix'.

The source code for this step is contained in the file `src/camera_calibration.py`

Calibration images are provided by the images `camera_cal/calibration*.jpg`
The calibration images are a set of 'chessboard' grid images. 

The first step consists of finding the intersections 'corners' in the calibration images using the OpenCV function `findChessboardCorners`. (_line 33_)
These (x,y) points are collected in the `imgpoints` array.

For each image, the `corners`, corresponding points representing the coordinate in real world space is added to to 
the `objpoints` array. 
Since the z coordinate is always zero, it does not need to be stored.
For each image, the object points are always the same because the same 'chessboard' is used in the calibration images.

The second step uses the OpenCV calibrateCamera function using the `objpoints` and `imgpoints` determined in the first step. (_line 58_)
The calibration results are tested using OpenCV `undistort`. (_line 71_)  
Example undistorted image is shown here:

![alt text][image1]

## Pipeline (single images)

### 1. Undistort Image
For video processing, as the first step, each image is undistorted with calibration results.

This is done in `src/image_utils.py` (_line 21_)  
An example of an undistorted image is shown here:
![alt text][image2]

### 2. Thresholded Binary Image
Thresholding an image according to some criteria is used to create a binary image that isolates the lane lines.  
A combination of gradient and color thresholds is used to generate the binary image.

This step is critical in the pipeline.
If the lane line cannot be detected and isolated well enough to represent the line, a good fit cannot be found.  
For example, if the look ahead filter algorithm fails to find a good fit, 
backing off and retrying the sliding windows algorithm will not help because the line pixels are not detected.

#### Color Spaces and Channels
An import aspect of finding the lane lines is using the right color space channel that represents the lane line well.  
For example, with RGB color space, Red has high values for lane line pixels, Green has moderate values, Blues has low to zero values.  
So using the Blue channel will do poorly to find lane lines.

A combination of HLS S channel, HSV V channel, and RGB Red channel is used for lane line pixel detection.  
HLS S does well for light backgrounds such as the bridge of concrete construction,  
well as RGB Red channel does better for dark backgrounds such as road surface following bridges, such as frame 613.

#### Sobel Gradient Thresholding
[Sobel gradient filters](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html)
can be used to detect edges (as an alternative to Hough transforms)

Sobel function code is in `src/sobel_utils.py`.  
It has functions for finding gradients in x and y directions `abs_sobel_thresh`,  
magnitude `mag_thresh` and direction `dir_threshold`.  
Magnitude and direction is used in conjunction, and it captures aspects of x and y direction as well.

#### Color Thresholding
For color thresholding, HLS S and HSV V channels (instead of Gray or Red) as they were found to provide good lane line detection
through experimentation.

Sobel gradient and Color channel thresholding were combined to generate binary images.
The code for this is in the function colorAndGradientThresholdBinary (_src/combined_binary_util.py:line 5_)

An example of a lane image processed through sobel and color thresholding is shown here:

![alt text][image3]

### 3. Perspective transformation to get flat birds-eye view of lane image

The perspective transformation matrix is calculated in file `src/perspective_transform_matrix.py`  
using the OpenCV function `getPerspectiveTransform` at _line 39_.  
The inverse perspective transformation matrix is calculated at _line 42_. It is used to warp warped images back to perspective view.

The source and destination points used by `getPerspectiveTransform` is selected as follows:

```
# Define 4 source points 
src_points = [[596,450], [685,450], [1100,720], [200,720]]
src = np.float32(src_points)

# Define 4 destination points
dst_points = [[320, 0], [960, 0], [960, 720], [320, 720]]
dst = np.float32(dst_points)
```

The perspective transform matrix from `getPerspectiveTransform` is employed by the function `image_warp` in `src/image_utils.py`.  
This function was tested on the `test_images/straight_lines1.jpg` image with the following result:

![alt text][image4]

### 4. Lane Line Detection and Polynomial Fitting
Two methods are available to detect lane line pixels:
1. Sliding Windows: histogram based approach
2. Look Ahead Filter: can be used if a prior fit is available.

#### Sliding Windows
Is implemented by function slidingWindowsPolyFit in `src/lane_fit_utils.py` (_line 109_)  
Algorithm approach is as follows:
1. finds the starting x coords for left and right lanes by finding the histogram peaks of non-zero pixels for left and right halves of the image.
2. divide the image into even sections along y-axis
3. starting from bottom, for each section:
  - find mean of non-zero pixels and set as current x value of line
  - adjust window with current x
4. fit non-zero pixel positions with numpy.polyfit as 2nd order polynomial
5. lane curvature radius is calculate via call to lane_curvature at line _line 211_

#### Look Ahead Filter
Is implemented by function lookAheadFilter in `src/lane_fit_utils.py` (_line 219_)  
Algorithm approach is as follows:
1. given prior fit, create region with some margin to either side of line
2. detect non-zero pixels in region
3. fit polynomial to non-zero pixels positions


Results for both approaches are shown for image `test_images/straight_lines1.jpg`,
where yellow lines denote polynomial fit lines.

##### Sliding Windows
![alt text][image5]

##### Look Ahead Filter
![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
### 5. Line curvature radius calculation and lane offset

#### Line Curvature
Is calculated via function line_curvature in `src/lane_fit_utils.py` (_line 43_)
Given a polynomial fit,  
x and y points are calculated for the image window for real world dimensions,  
these points are refit to another polynomial,  
radius is then calculated using the real world polynomial fit.

#### Lane offset
Is calculated via function lane_offset in `src/lane_fit_utils.py` (_line 70_)
Given left and right fit  
calculate the x coord for left and right lines at image point closest to vehicle,  
calculate difference between mid point of left and right x coords and image mid point  
convert to real world dimensions

### 6. Example of lane overlay on road image
Plotting lane overlay on lane image is implemented in function `plot_lane` in file `src/lane_finder.py` (_line 70_)
Basically the line fit is drawn on warped image,  
the warped image is un-warped using inverse perspective transform matrix,  
the un-warped lane image is then overlayed onto lane image.

An example result is shown here:
![alt text][image7]

---

### Pipeline (video)

####1. [link to youtube video result](https://youtu.be/zp9oSdVjiws)

<iframe width="560" height="315" src="https://www.youtube.com/embed/zp9oSdVjiws" frameborder="0" allowfullscreen></iframe>

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

##### Lane line detection

Finding lane line pixels was critical for entire pipeline.
A color space channel may work well for certain frame sections like dark road surfaces, but not well for light road surfaces.
A combination of multiple channels from HLS, HSV and RGB color spaces was used to address different situations.

Distinguishing from proximate lines like bridge interface was problematic,
finding the thresholding limits that works was empirical.

#### 2. Where will your pipeline likely fail?  
Vehicles in the lane ahead would confuse the current pipeline implementation.  

Lane line discontinuation for extended frames would cause hiccups.  
Tight curves would require faster radius updating.  
More shadows could be problematic.

#### 3. What could you do to make it more robust?

The techniques employed seems brittle and requires fine-tuning.  
Techniques that are adaptive to frame image characteristics could be explored,
e.g. use different color space/channels if image lighter/darker.

Could explore use of behavioral-cloning techniques in conjunction.



