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

[//]: # (Image References)

[image1]: ./output_images/undistorted_test_image.jpg "Undistorted"
[image2]: ./output_images/undistorted.jpg "undistorted_2"
[image3]: ./output_images/originalSobelCombinedGradients.png "Sobel Binary Example"
[image4]: ./output_images/originalColorAndSobelCombinedThresholds.png "Color Sobel Binary Combined"
[image5]: ./output_images/binaryWarped.png " binary warped"
[image6]: ./output_images/SlidingWindow.png "Sliding Window"
[image7]: ./output_images/finalOutputInCurve.png " finish"
[image8]: ./output_images/SystemDesignCalibration.jpg " System Design"
[image9]: ./output_images/SystemDesignImageProcess.jpg " System Design"
[image10]:./output_images/sobelBinaryImagesAllThree.png "Sobel binary"
[image11]:./output_images/colorThresholdsAll.png "Sobel binary"
[image12]:./output_images/combinedColorBinary.png "Color binary"
[image13]:./output_images/cobinedColorSobelBinary.png "Sobel binary"
[image14]:./output_images/binaryWarpedUnmasket.png "warped binary unmasked"
[image15]:./output_images/binaryWarped_final.png "warped binary unmasked"
[image16]: ./output_images/slidingWindow_1.png "Sliding Window"
[video1]: ./submission_video.mp4 "Submission Video"


 ### Source
 [All used code for this submission is located in this file advancedLaneLinesFindingSecodLevel.ipynb](./advancedLaneLinesFindingSecodLevel.ipynb)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

IN[3]:
```python
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```  

Further I create a list of the calibration image with the help of glob module.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:  

![Undistorted][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image  Done in [235].

```python
# Load calibration data generated in calibrate_camera
    mtx, dist = pickle_load()
    # Calculate the undistorted image
    image_undistored = cv2.undistort(img, mtx, dist, None, mtx)
```
![undistorted_2][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For getting the almost best gradient and color space anlysis I spend a lot of time to play around with different color spaces like LAB,RGB,HLS and LUV and thresholds. The key is to plot the images after each creation of a binary image. So if one makes i.e. abs sobel_x and sobel magnitude--> plot after each gradient process to evaluate the influence of each process. Now one got a feeling which process performs in a special area of the image better than others. Once we know enough about the influences we are able to implement a fitting condition algorithm to combine the different binary images. The rule is: Take an OR condition if the images complement each other and take an AND if u want to skip out bad areas out of one image. The same for color combination of color binary images.

In [218] the sobel function calls In [167] the functions definition.
```python
grad_x_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 255))
grad_y_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(65, 255))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(65, 255))
```
Of course there is a lot potential in define better thresholds but that could be a n endless game. So, for the difficult images it was sufficient. Below is shown how I combined the different binary threshhold results to get the best out of them.

```python
# Combine the thresholds
combined = np.zeros_like(grad_x_binary)
combined[((grad_x_binary == 1)&(mag_binary ==1)&(grad_y_binary == 1)) | ((grad_x_binary == 1) & (mag_binary ==1))| (grad_x_binary == 1) ] = 1
```
Reults of each binary Image:

![sobel binary images][image10]

What do we see: Sobel_x is showing a huge pixels and is the best gradient in a difficult image like the test5 image. So sobel_x is the only gradient in the combination algo which stand alone like | (grad_x_binary == 1). Further I used a AND condition between  ((grad_x_binary == 1) & (mag_binary ==1)) for skipping out some negative noise areas in the mag_binary image. And on. It is all about try and catch. For sure out of scope there are mathematically possibilities to generalize a method for calculate the optimal thresholds and also the combination. But that would be out of  project scope.

What I did for creating color channel binary images:
In [219] I used HLS, LAB and LUV for getting the best combination of all color channel strength. For figure out the thresholds I played around with the test5 image In [227]
* LAB b_thresh(25,255) -- LUV l_thresh(85,255) -- HLS s_thresh(180,255)

```python
# Call of combination
combined_binary[((s_binary == 1) & (b_binary == 1))|((l_binary == 1)&(s_binary == 1))|((l_binary == 1)|(b_binary == 1))] = 1 
```      
Result of each channel from the different color spaces:

![color channel binary images][image11]

Result of color channel combination:

![combined color binary images][image12]

So and at the end naturally we need to combine the two binary cathegories sobel and color with an OR statement to complement each other.

![Sobel Binary Example][image13]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
In [235]
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
Especially I chose hard coded src and destination point in the following manner  definition In [222]:

```python
## Define 4 source points
    src = np.float32([[180, img.shape[0]], [575, 460], 
                      [705, 460], [1150, img.shape[0]]])
    # Define 4 destination points
    dst = np.float32([[320, img.shape[0]], [320, 0], 
                      [960, 0], [960, img.shape[0]]])
```

This is the output of a original warped test image and a binary warped:

![binary warped unmasked][image14]

For skip the noise out of the picture I use a masking method (definition in [13])
```python
 # Define image mask (polygon of interest)
    binaryWarpedImageShape = binary_warped.shape
    vertices = np.array([[(200, binaryWarpedImageShape[0]), (200, 0), (binaryWarpedImageShape[1] - 200, 0), 
                      (binaryWarpedImageShape[1]-200, binaryWarpedImageShape[0])]], dtype=np.int32)
    binaryWarpedMaskedImage = region_of_interest(binary_warped, vertices)

```
Warped image after region masking

![binary warped unmasked][image15]

In my opinion it is the most significant influence to the project result to calculate very, very well binary images. The pipeline for this challange video is secondary. Sanity checks are not necessary in this video like I figured out and show later in this writeUp with the submission video.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find out where the line pixels are I use the recommended  slidingWindowMethod() method in [234]. Input arguments are the masked binary image shown above and the leftx_base, rightx_base. For the base positions I applied a little sanity check to puffer an old base value if the current cycle frame is too bad.

```python
if (abs(leftx_base - rightx_base) < 250) & (abs(leftx_base - rightx_base) > 160):
    leftx_base_old = leftx_base
    rightx_base_old = rightx_base
```

The image shows the binary masked image, the 9 sliding windows for detect lines and also in green the fitted polynomial in the background.

![Sliding Window][image16] 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
TO calculate the deviation to center I did the following in [235]:

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

To calculate the radius there is a function called convertRadiusIntoMeter(ploty,y_eval,leftx, lefty, rightx, righty) in [18]: The main part is to define a transformation convention: How many pixels are one meter in x or y dimension.

```python
# Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension    
    # Fit a second order polynomial to each
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Definition of plot fucntion in [19] def drawPolynomialsBackIntoOriginalImage. Here is an example of my result on a difficult section:

![alt text][image7]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my submission video](./submission_video.mp4)

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

In general my pipeline should work like this:

1. Gerate a calibration pickle set. I do the with the calibration chessboard set of the repository. 
 This step is needed only once and at the end I created a wide_dist_pickle file where the camera matrix and the dist are stored. The main part here is implemented in the find2D3DCorners()[3].There is a single path on the leften side in my system design overview shown below.

 ![alt text][image8]

2. I chose to get things running and build up a pipeline out of the code snippets from the lessons. So there are two pipelines shown in the image of processing below.
In one option I only use sliding window mehtod and one with some checks about the quality of line detection after sliding window method. My plan was to implement first a stright piepeline without any sanity checks and make a proof of concept for one picture and also the video pipeline. One can see it on the righten side in the system design image if you just ignore the if conditions after the block of sanity check and the block of look ahead filter.

 ![alt text][image9]

Further more there is also a common part for the first four steps including:
* undistortion of the input frames
* Apply sobel and color threshold methods and combine the results
* Calc source and destination points and make a perspective tranform
* Apply the sliding window method to detect line pixelx of the left and right line
* also the calculation of radius and deviation is the same

The result was that everytime in the curve with hard shadows and bad lanes markings I got a problem to detect lines. After good feedback I adapt the threshold and add two more color spaces for calculate better binary results on test image 5.

3. The additional color spaces and the serious tuning of sobal and color thresholds and also the improved logic to combine makes the pipeline very, very robust during the whole project video. Extract the basex checks before call of sliding window method, there were no more sanity checks needed anymore.  


# What did I additionaly try to implement? 

## Sanity checks: 

Did we get enough points to fit a polynomial? Avoid empty array as input to np.polifit method.

```python
if (len(leftx) < 1500):
   leftx = leftx_old 
   lefty = lefty_old
   detected == False
else:
   leftx_old = leftx
   lefty_old = lefty
if (len(rightx) < 1500):
   rightx = rightx_old
   righty = righty_old
   detected == False
```

So everytime we detect good lines we want to use the look ahead method in[26]. If sanity checks fails the pipeline shall use the sliding window method as long as we have good results of lines again.

Therefore I chose to try from scipy.stats.stats import pearsonr () and add a sensful threshold.

* pearsonr() returns a two-tuple consisting of the correlation coefficient and the corresponding p-value:

* The correlation coefficient can range from -1 to +1. The null hypothesis is that the two variables are uncorrelated. The p-value is a number between zero and one that represents the probability that your data would have arisen if the null hypothesis were true.

Take a look into sanity check function in [14]:
```python
    ret_left = pearsonr (left_fitx_old, left_fitx)
    ret_right = pearsonr (right_fitx_old, right_fitx)
    
    if (ret_left[0] > threshold):
        left_fitx_old = left_fitx
        detect= True   
    else:
        left_fitx = left_fitx_old
        detect= False    
    if (ret_right[0] > threshold):
    ...[]
```

For some reason I did not figured out yet, the pipeline did not working well showed frozen polyfill pixels during the video. So now we need not to be real time and under this point of view there is no necessity to use the lookAheadFilter.

## Object orientated approach

I wanted to use the advantages of classes. My python knowledge was not enough to deal with classes and because of that and other missing programming skills I did not spend more time on it for the submission. But it will be a good challeang for me to implement the pipeline as object oreientated. So the disadvantage of global variables is hard :-) but it works for me;-)...with some workarounds ;-).

The OOP experiences stuff is shown in the files below:
* [OOP Pipeline](./advancedLaneLinesFindingTryObjectOrientated.ipynb)
* [line.py Class](./line.py)
* [LineDetection.py Class](./LineDetection.py)
* [functions pynthon file with functions](./fucntions.py)

But I did not come around and time was running. If u want u can take a look. So I decided to get an acceptable run with the implementation of advancedLaneLinesFindingSecodLevel.ipynb which produce the linked output video above.

My understanding of what is the sense of using the collection module is very well but I did not get it running as quick as I should.

Thats it and I learned a lot about problems during lane line detection and what are possibilies to deal with them. For example store good lines in a collection and use the avarage or use sliding window only if necessary. But at the end I struggle on python :-)











