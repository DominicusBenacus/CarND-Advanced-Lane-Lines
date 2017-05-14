## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_test_image.jpg "Undistorted"
[image2]: ./output_images/undistorted.jpg "undistorted_2"
[image3]: ./output_images/originalSobelCombinedGradients.png "Sobel Binary Example"
[image4]: ./output_images/originalColorAndSobelCombinedThresholds.png "Color Sobel Binary Combined"
[image5]: ./output_images/binaryWarped.png " binary warped"
[image6]: ./output_images/SlidingWindow.png " Sliding Window"
[image7]: ./output_images/finish.png " finish"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

```python
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```  

Further I create a list of the calibration image with the help of glob module

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:  

![Undistorted][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![undistorted_2][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds and sobel gradient thresholds. I combined all three sobel gradient calculation and mixed it.

Sat first I combined sobel absolute direction and mesh for get a good mix:


![Sobel Binary Example][image3]
![Color Sobel Binary Combined][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
calcSrcAndDstPoints
 The warp process takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points: 

```python
    # using cv2.findChessboardCorners
    src, dst = calcSrcAndDstPoints(combinedThresholdsBinaryImage)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Warp the image using OpenCV warpPerspective()
    img_size = (combinedThresholdsBinaryImage.shape[1], combinedThresholdsBinaryImage.shape[0])
    binary_warped = cv2.warpPerspective(combinedThresholdsBinaryImage, M, img_size, flags=cv2.INTER_LINEAR) 
```
Especially I chose hard coded src and destination point in the following manner:

```python
## Define 4 source points
    src = np.float32([[180, img.shape[0]], [575, 460], 
                      [705, 460], [1150, img.shape[0]]])
    # Define 4 destination points
    dst = np.float32([[320, img.shape[0]], [320, 0], 
                      [960, 0], [960, img.shape[0]]])
```

This is the output: 
![binary warped][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TO find out where the line pixels are I use the recommended mehtod slidingWindowMethod().

This is my Output

![Sliding Window][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
TO calculate the deviation to center I did the following:

```python
# Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    mid_of_lane = leftx_base + (rightx_base - leftx_base) / 2
    car_pos = binary_warped.shape[1] / 2
    deviation = int(abs(mid_of_lane - car_pos) * (3.7 / 700) * 100)  
```

To calculate the radius there is a fucntion called convertRadiusIntoMeter(ploty,y_eval,leftx, lefty, rightx, righty) in my .jypnb

snipped:
```python
# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension    
    # Fit a second order polynomial to each
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

- So I chose to make thing running and build up a pipeline out of the snippets from the lessons.
- So there are two pipelines. Ine only use sliding window mehtod and one with some checks about the quality of line detection after sliding window method. My plan was to implement first a stright piepeline without any sanity check and make a proof of concept for one picture and also the video pipeline. The result was that everytime in the curve with hard shadows and bad lanes I got a problem to detect lines.

Thats ok because I expected it and start to implement the second level of the pipeline. This means to add some sanity checks.

first step. detect enough points to fit a lane. If not take the old ones. Therfore I used global variables like leftx_old
```python

if (len(leftx) < 1500):
        leftx = leftx_old 
        lefty = lefty_old
        detected = False
    else:
        leftx_old = leftx
        lefty_old = lefty
    if (len(rightx) < 1500):
        rightx = rightx_old
        righty = righty_old
        detected = False
    else:
        rightx_old = rightx
        righty_old = righty
``` 

Ok now we chcked that. but even if we detect enough points.Wahta bout quality of that points.
Therefore I chose to try from scipy.stats.stats import pearsonr () and add a sensful threshold.

* pearsonr() returns a two-tuple consisting of the correlation coefficient and the corresponding p-value:

* The correlation coefficient can range from -1 to +1. The null hypothesis is that the two variables are uncorrelated. The p-value is a number between zero and one that represents the probability that your data would have arisen if the null hypothesis were true.

So it looks inside the sanity check function:

```python
    ret_left = pearsonr (left_fitx_old, left_fitx)
    ret_right = pearsonr (right_fitx_old, right_fitx)
    
    if (ret_left[0] > threshold):
        left_fitx_old = left_fitx   
    else:
        left_fitx = left_fitx_old
    
    if (ret_right[0] > threshold):
    ...[]
```

Ok now we check more and filer out bad line detection for a short tim horizon.

Further the plan was to create an attribute detect lines = False or something like this. That atttribute allows to decide whether choose the sliding window method or call the lookAhead() function which is more effective once we detect good lines.



