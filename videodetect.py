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
import numpy as np
from moviepy.editor import VideoFileClip


def processing_image(image) :
	src_p = np.float32([[92, 1532], [1926, 1532], [776, 918], [1314, 918]])
	dst_p = np.float32([[92, 1532], [1926, 1532], [92, 746], [1926, 746]])
	# src_p = np.float32([[272, 1479], [1828, 1478], [660, 1100], [1455, 1100]])
	# dst_p = np.float32([[272, 1479], [1828, 1478], [270, 678], [1828, 678]])
	M = cv2.getPerspectiveTransform(src_p, dst_p)
	
	warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))


	gradF = F.gradFilters(_sobel_x_thresh=(30, 100), 
	                _sobel_y_thresh=(30, 100), 
	                _mag_thresh=(30, 100), 
	                _dir_thresh=(0.0, 0.4))

	colorF = F.colorFilter()
	_, _ = gradF.compute_mag_grad(warped_image, 3)

	xFilter = gradF.getXgradFilter()
	yFilter = gradF.getYgradFilter()

	colorFilter = colorF.getColorFilter(warped_image, (100, 255))

	combine_Bin_image = np.zeros_like(colorFilter)
	combine_Bin_image[(((xFilter == 1) & (yFilter ==1))==1) | (colorFilter == 1)] = 1

	Lfinder = FLane.Lane()

	Lfinder.processing(combine_Bin_image, warped_image, _visualization = True)


	inv_M = cv2.getPerspectiveTransform(dst_p, src_p)
	sign_img = np.ones_like(Lfinder.result) * 255
	inv_sign_img = cv2.warpPerspective(sign_img, inv_M, (image.shape[1], image.shape[0]))
	inv_warped_image = cv2.warpPerspective(Lfinder.result, inv_M, (image.shape[1], image.shape[0]))
	image[(inv_sign_img == 255)] = inv_warped_image[(inv_sign_img == 255)]


	# cover = np.zeros_like(xFilter)
	# cover[:, 130:1200] = 1
	# xFilter[(cover == 0)] = 0
	# yFilter[(cover == 0)] = 0
	# colorFilter[(cover == 0)] = 0


	
	# text = 'on the center.'

	# if (diff_dis > 0) :
	#     text = 'right of center : %f m'%(diff_dis)
	# elif (diff_dis < 0) :
	#     text = 'left of center : %f m'%(-diff_dis)
	    
	# cv2.putText(image, ('Radius of Curvature : %d m'%(np.int(mean_r))), (50,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2, cv2.LINE_AA)
	# cv2.putText(image, text, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2, cv2.LINE_AA)



	return image


output_dir = 'output_images/video4.mp4'

print(os.path.isfile('./data/video/video.mp4') )

clip1 = VideoFileClip('./data/video/video.avi').subclip(0,10)

clip = clip1.fl_image(processing_image)
clip.write_videofile(output_dir, audio=False)



