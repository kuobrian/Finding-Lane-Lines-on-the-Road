# -*- coding: utf-8 -*-
import os
import sys
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import glob
import src.CameraCalibration as cc
import src.Filters as F
import src.FindingLane as FLane
from time import sleep
from threading import Thread
import numpy as np

np.seterr(divide='ignore',invalid='ignore')

def close(time):
    sleep(time)
    plt.close()


if __name__ == "__main__":
	# calibration = cc.Calibration('./camera_cal', 'jpg', 9, 6)
	# calibration.cameraCalibration(_drawCorner = False)
	
	# calibration.undistort_list(calibration.getDataList())
	test_img_list = glob.glob('./test_images/*.jpg')
	
	src_p = np.float32([[92, 1532], [1926, 1532], [776, 918], [1314, 918]])
	dst_p = np.float32([[92, 1532], [1926, 1532], [92, 746], [1926, 746]])
	M = cv2.getPerspectiveTransform(src_p, dst_p)

	# # compute warp images
	# fig_und = plt.figure(figsize=(24, 9))
	# fig_und.suptitle('undistort image (left) & warp result (right)', fontsize=20)
	# for i, _filename in enumerate(test_img_list) :
	# 	img = cv2.cvtColor(cv2.imread(_filename), cv2.COLOR_BGR2RGB)
	# 	# und_img = calibration.undistort(img)
	# 	warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
	# 	und = fig_und.add_subplot(4, 4, 2 * i + 1)
	# 	und.imshow(img)
	# 	und = fig_und.add_subplot(4, 4, 2 * i + 2)
	# 	und.imshow(warped)
	# plt.show()


	gradF = F.gradFilters(_sobel_x_thresh=(40, 120), 
	                    _sobel_y_thresh=(30, 100), 
	                    _mag_thresh=(30, 100), 
	                    _dir_thresh=(0.0, 0.4))
	colorF = F.colorFilter()
	
	for i, _filename in enumerate(test_img_list) :	
		fig_und = plt.figure(figsize=(24, 9))
		fig_und.suptitle('warped image & grad result (x, y, mag., dir., color, combine(x, color))', fontsize=20)
	
		img = cv2.cvtColor(cv2.imread(_filename), cv2.COLOR_BGR2RGB)
		# und_img = calibration.undistort(img)
		warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
		
		_, _ = gradF.compute_mag_grad(warped, 3)
		_ = gradF.compute_dir(warped, 21)

		xFilter = gradF.getXgradFilter()
		yFilter = gradF.getYgradFilter()
		magFilter = gradF.getMagFilter()
		dirFilter = gradF.getDirFilter()
		
		colorFilter = colorF.getColorFilter(warped, (120, 255))
		
		combine = np.zeros_like(colorFilter)
		combine[(xFilter == 1) | (colorFilter == 1)] = 1
		# plt.imshow(combine)
		# plt.show()
		
		finder = FLane.Lane()
		finder.processing(combine, warped, _visualization = True)
		# plt.imshow(finder.result)
		# plt.show()

		inv_M = cv2.getPerspectiveTransform(dst_p, src_p)
		sign_img = np.ones_like(finder.result) * 255
		inv_warped = cv2.warpPerspective(finder.result, inv_M, (img.shape[1], img.shape[0]))
		inv_sign_img = cv2.warpPerspective(sign_img, inv_M, (img.shape[1], img.shape[0]))



		cover = np.zeros_like(xFilter)
		cover[:, 0:1950] = 1
		cover[:, 750:1250] = 0
		# plt.imshow(cover, cmap='gray')
		# plt.show()

		colorFilter_cp = np.copy(colorFilter)
		colorFilter_cp[(cover == 0)] = 0

		processed_img = cv2.cvtColor(cv2.imread(_filename), cv2.COLOR_BGR2RGB)
		processed_img[(inv_sign_img == 255)] = inv_warped[(inv_sign_img == 255)]

		# plt.imshow(processed_img)
		# plt.show()
		# assert(0)

		und = fig_und.add_subplot(2, 3, 1)
		und.imshow(img)
		
		und = fig_und.add_subplot(2, 3, 2)
		und.imshow(warped)

		und = fig_und.add_subplot(2, 3, 3)
		und.imshow(finder.result)

		und = fig_und.add_subplot(2, 3, 4)
		und.imshow(colorFilter_cp)

		und = fig_und.add_subplot(2, 3, 5)
		und.imshow(processed_img)

		# und = fig_und.add_subplot(2, 3, 6)
		# plt.title("combine")
		# und.imshow(combine, cmap='gray')

		# # thread1 = Thread(target=close, args=(3,))

		# # thread1.start()
		plt.show()
		# # assert(0)
	