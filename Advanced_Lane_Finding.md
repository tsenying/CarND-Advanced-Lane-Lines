
# Advanced Lane Finding Project #

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Project Files ###

- Project Report: Advanced_Lane_Finding.md (this file)
- Source Code: in `./src` directory
- Supporting Images: in `./output_images` directory
- Camera calibration data: `camera_cal/calibration_pickle.p`
- Perspective transformation data: `camera_cal/perspective_transform_pickle.p`
- Jupyter notebook used for initial investigations: `AdvancedLaneFinding.ipynb`
- Final video with lane overlay: `lane_video.mp4` (not in git repository, uploaded to https://youtu.be/zp9oSdVjiws)

### Running source files ###
1. generate camera calibration: `python src/camera_calibration.py`
2. generate perspective transformation matrices: `python src/perspective_transform_matrix.py`
3. process video file: `src/process_video_images.py`

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.jpg "Camera Calibration"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

###Camera Calibration

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

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

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

