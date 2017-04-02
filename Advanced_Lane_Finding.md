
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
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Camera Calibration

####1. Camera calibration is done using [OpenCV Camera Calibration support](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration)

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

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
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

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

