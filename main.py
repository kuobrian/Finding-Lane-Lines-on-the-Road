# -*- coding: utf-8 -*-
import os
import sys
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import glob
from time import sleep
from threading import Thread
from curve import Curves
from birdseye import BirdsEye
from lanefilters import LaneFilter
from utils import *
import numpy as np
from scipy.misc import imresize
from config import *


def lane_filter_test(path, laneFilter):
  img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
  binary = laneFilter.apply(img)
  masked_img = np.bitwise_and(birdsEye.sky_view(img), roi(birdsEye.sky_view(img)))
  masked_lane = np.logical_and(birdsEye.sky_view(binary), roi(binary))
  sobel_img = birdsEye.sky_view(laneFilter.sobel_breakdown(img))
  color_img = birdsEye.sky_view(laneFilter.color_breakdown(img))
  show_images([masked_img, color_img, sobel_img, masked_lane], per_row = 4, per_col = 1, W = 20, H = 5)

def curve_test(path, laneFilter):
  img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
  binary = laneFilter.apply(img)
  masked_lane = np.logical_and(birdsEye.sky_view(binary), roi(binary)).astype(np.uint8)
  result = curves.fit(masked_lane)
  display_image(result['image'])
 
def display_image(img, name="image"):
 	img = cv2.resize(img, (400, 300), interpolation=cv2.INTER_CUBIC)
	cv2.imshow(name,img) 
 	cv2.waitKey(0)
	cv2.destroyAllWindows()

def pipeline_debug(img):

  # ground_img = birdsEye.undistort(img)
  ground_img = img
  birdseye_img = birdsEye.sky_view(img)
    
  binary_img = laneFilter.apply(ground_img)
  sobel_img = birdsEye.sky_view(laneFilter.sobel_breakdown(ground_img))
  color_img = birdsEye.sky_view(laneFilter.color_breakdown(ground_img))
  
  wb = np.logical_and(birdsEye.sky_view(binary_img), roi(binary_img)).astype(np.uint8)
  result = curves.fit(wb)
    
  left_curve =  result['pixel_left_best_fit_curve']
  right_curve =  result['pixel_right_best_fit_curve']
    
  left_radius =  result['left_radius']
  right_radius =  result['right_radius']
  curve_debug_img = result['image']
  # words = result['vehicle_position_words']
  projected_img = birdsEye.project(ground_img, binary_img, left_curve, right_curve)
    
  return birdseye_img, sobel_img, color_img, curve_debug_img, projected_img, left_radius, right_radius

def verbose_pipeline(img):
  b_img, s_img, co_img, cu_img, pro_img, lr, rr = pipeline_debug(img)
  print(pro_img.shape)
  h, w = pro_img.shape[0], pro_img.shape[1]
  b_img = imresize(b_img, 0.25)
  s_img = imresize(s_img, 0.25)
  co_img = imresize(co_img, 0.25)
  cu_img = imresize(cu_img, 0.25)
  
  width, height = b_img.shape[1], b_img.shape[0]
  offset = [0, width*1, width*2, width*3]
  pro_img[:height, int(offset[0]): int(width), :] = b_img
  pro_img[:height, offset[1]: offset[1] + width] = co_img
  pro_img[:height, offset[2]: offset[2] + width] = s_img
  pro_img[:height, offset[3]: offset[3] + width] = cu_img

  # text_pos = "vehicle pos: " + pos
  text_l = "left r: " + str(np.round(lr, 2)) 
  text_r = " right r: " + str(np.round(rr, 2))
    
  cv2.putText(pro_img, text_l, (20, height+80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 2,  cv2.LINE_AA)
  cv2.putText(pro_img, text_r, (1000, height+80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 2,  cv2.LINE_AA)
  # cv2.putText(pro_img, text_pos, (620, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
  return pro_img



def pipeline_test(path):
  img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
  birdseye_img, sobel_img, color_img, curve_debug_img, projected_img, left_radius, right_radius = pipeline_debug(img)
  print("left radius:", left_radius, "m |", "right radius:", right_radius, "m")
  # print(words)
  show_images([birdseye_img, sobel_img, color_img], per_row = 3, per_col = 1, W = 15, H = 3)
  show_images([curve_debug_img, projected_img], per_row = 3, per_col = 1, W = 15, H = 3)

if __name__ == "__main__":

	# calibration = cc.Calibration('./camera_cal', 'jpg', 9, 6)
	# calibration.cameraCalibration(_drawCorner = False)
	# calibration.undistort_list(calibration.getDataList())

	birdsEye = BirdsEye(src_p, dst_p)
	laneFilter = LaneFilter(p)
	curves = Curves(number_of_windows = 9, margin = 100, minimum_pixels = 50, 
                ym_per_pix = 30 / 720 , xm_per_pix = 3.7 / 700)
	
	for i, filename in enumerate(test_img_list):
		img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
		# warped = birdsEye.sky_view(img)
		
		# blur_ksize = 5  # Gaussian blur kernel size
		# blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
		# canny_lthreshold = 50  # Canny edge detection low threshold
		# canny_hthreshold = 150  # Canny edge detection high threshold
		# edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)

		# lane_filter_test(filename, laneFilter)
		# curve_test(filename, laneFilter)
		# pipeline_test(filename)
		
		plt.imshow(verbose_pipeline(img))
		plt.show()
		
	