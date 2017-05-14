import numpy as np
from line import Line
import functions
import cv2                         # openCV
import glob                        # Filename pattern matching
import matplotlib.pyplot as plt    # 2D plotting
import matplotlib.image as mpimg
import pickle
import os

class LineDetection(object):
    """
    Tracks the lane in a series of consecutive frames.
    """

    def __init__(self, first_frame):
        """
        Initialises a tracker object.
        Parameters
        ----------
        first_frame     : First frame of the frame series. We use it to get dimensions and initialise values.
        n_windows       : Number of windows we use to track each lane edge.
        """
        (self.h, self.w, _) = first_frame.shape
        self.left = None
        self.right = None        
        self.initialize_lines()

    def initialize_lines(self):
        
        
        # Create empty lists to receive left and right lane pixel indices
        l_indices = np.ones([0], dtype=np.int)
        r_indices = np.ones([0], dtype=np.int)
        
        self.left = Line(x=l_indices, y=l_indices, h=self.h, w = self.w)
        self.right = Line(x=r_indices, y=r_indices, h=self.h, w = self.w)

    def process(self, img):
        """
        Performs a full lane tracking pipeline on a frame.
        Parameters
        ----------
        frame               : New frame to process.
        
        Returns
        -------
        Resulting frame.
        """
        # Load calibration data generated in calibrate_camera
        mtx, dist = functions.pickle_load()
        # Calculate the undistorted image
        image_undistored = functions.cv2.undistort(img, mtx, dist, None, mtx)
        
        #Calculate combined binaryImage based on a mix of both
        # combinedColorThresholds and combinedSobelGradient threshold methods
        combinedThresholdsBinaryImage = functions.combineColorAndGradientThresholds(image_undistored)
        # define source_img and destination_img point for     preparing the perspective transform
        # using cv2.findChessboardCorners
        src, dst = functions.calcSrcAndDstPoints(combinedThresholdsBinaryImage)
        # Given src and dst points, calculate the perspective transform matrix
        M = functions.cv2.getPerspectiveTransform(src, dst)
        
        # Warp the image using OpenCV warpPerspective()
        img_size = (combinedThresholdsBinaryImage.shape[1], combinedThresholdsBinaryImage.shape[0])
        binary_warped = functions.cv2.warpPerspective(combinedThresholdsBinaryImage, M, img_size, flags=cv2.INTER_LINEAR)    
        
        # Define image mask (polygon of interest)
        binaryWarpedImageShape = binary_warped.shape
        vertices = np.array([[(200, binaryWarpedImageShape[0]), (200, 0), (binaryWarpedImageShape[1] - 200, 0), 
                          (binaryWarpedImageShape[1]-200, binaryWarpedImageShape[0])]], dtype=np.int32)
        binaryWarpedMaskedImage = functions.region_of_interest(binary_warped, vertices)
        
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binaryWarpedMaskedImage[binaryWarpedMaskedImage.shape[0]/2:,:], axis=0)    
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        mid_of_lane = leftx_base + (rightx_base - leftx_base) / 2
        car_pos = binary_warped.shape[1] / 2
        deviation = int(abs(mid_of_lane - car_pos) * (3.7 / 224) * 100) 
        
        out_img, leftx, lefty, rightx, righty = functions.slidingWindowMethod(binaryWarpedMaskedImage,leftx_base, rightx_base)
        
        
        
        self.left.process_points(leftx, lefty)
        self.right.process_points(rightx, righty)
        
        left_fitx = self.left.get_points()
        right_fitx = self.right.get_points()   
        
        left_curveradMeter = self.left.radius_of_curvature()
        right_curveradMeter = self.right.radius_of_curvature()
        
        #sanity check --> TODO   
        
        #left_fit, right_fit = polynomFit2nd(lefty, leftx, righty, rightx)
    
        #left_fitx, right_fitx, ploty = genrateValuesXYforPlot(binary_warped,left_fit,right_fit)
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        
        #left_curveradPixelSpace, right_curveradPixelSpace = getRadiusPixelSpace(left_fit,right_fit,y_eval)
        
        #left_curveradMeter, right_curveradMeter = convertRadiusIntoMeter(ploty,y_eval,leftx, lefty, rightx, righty)
    
        
        #Compute the inverse perspective transform to unwarped the image
        Minv = functions.cv2.getPerspectiveTransform(dst, src)
        
        finalOutputImage = functions.drawPolynomialsBackIntoOriginalImage(binaryWarpedMaskedImage,image_undistored, out_img, deviation,left_fitx, right_fitx, left_curveradMeter, right_curveradMeter, ploty, Minv)
    
        return finalOutputImage