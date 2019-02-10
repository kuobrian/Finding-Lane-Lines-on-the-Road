# -*- coding: utf-8 -*-
import os
import sys
import cv2
from utils import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import glob
import src.CameraCalibration as cc
import src.Filters as F
import src.FindingLane as FLane


if __name__ == "__main__":
	calibration = cc.Calibration('./camera_cal', 'jpg', 9, 6)
	calibration.cameraCalibration(_drawCorner = True)
	
	# calibration.undistort_list(calibration.getDataList())
	test_img_list = glob.glob('./test_images/*.jpg')
	
	fig_und = plt.figure(figsize=(24, 9))
	fig_und.suptitle('original image (left) & undistort result (right)', fontsize=20)
	for i, _filename in enumerate(test_img_list) :
		img = cv2.cvtColor(cv2.imread(_filename), cv2.COLOR_BGR2RGB)
		und_img = calibration.undistort(img)
		und = fig_und.add_subplot(4, 4, 2 * i + 1)
		und.imshow(img)
		und = fig_und.add_subplot(4, 4, 2 * i + 2)
		und.imshow(und_img)
	# plt.show()

	# src_p = np.float32([[535, 490], [770, 490], [1090, 660], [290, 660]])
	src_p = np.float32([[535, 490], [770, 490], [1100, 719], [190, 719]])
	dst_p = np.float32([[320, 0], [1000, 0], [1000, 720 - 1], [320, 720 - 1]])
	M = cv2.getPerspectiveTransform(src_p, dst_p)
	# print (test_img_list[4])
	# img = cv2.cvtColor(cv2.imread(test_img_list[6]), cv2.COLOR_BGR2RGB)
	# und_img = calibration.undistort(img)
	# warped = cv2.warpPerspective(und_img, M, (img.shape[1], img.shape[0]))
	# plt.imshow(warped)
	# f = plt.show()

	fig_und = plt.figure(figsize=(24, 9))
	fig_und.suptitle('undistort image (left) & warp result (right)', fontsize=20)
	for i, _filename in enumerate(test_img_list) :
		img = cv2.cvtColor(cv2.imread(_filename), cv2.COLOR_BGR2RGB)
		und_img = calibration.undistort(img)
		warped = cv2.warpPerspective(und_img, M, (img.shape[1], img.shape[0]))
		und = fig_und.add_subplot(4, 4, 2 * i + 1)
		und.imshow(und_img)
		und = fig_und.add_subplot(4, 4, 2 * i + 2)
		und.imshow(warped)
	# f = plt.show()



	gradF = F.gradFilters(_sobel_x_thresh=(30, 100), 
                    _sobel_y_thresh=(30, 100), 
                    _mag_thresh=(30, 100), 
                    _dir_thresh=(0.0, 0.4))

	colorF = F.colorFilter()

	fig_und = plt.figure(figsize=(24, 18))
	fig_und.suptitle('warped image & grad result (x, y, mag., dir., color, combine(x, color))', fontsize=20)
	for i, _filename in enumerate(test_img_list) :
		img = cv2.cvtColor(cv2.imread(_filename), cv2.COLOR_BGR2RGB)
		und_img = calibration.undistort(img)
		warped = cv2.warpPerspective(und_img, M, (img.shape[1], img.shape[0]))
		
		_, _ = gradF.compute_mag_grad(warped, 3)
		_ = gradF.compute_dir(warped, 21)

		xFilter = gradF.getXgradFilter()
		yFilter = gradF.getYgradFilter()
		magFilter = gradF.getMagFilter()
		dirFilter = gradF.getDirFilter()
		
		colorFilter = colorF.getColorFilter(warped, (120, 255))
		
		combine = np.zeros_like(colorFilter)
		combine[(xFilter == 1) | (colorFilter == 1)] = 1
		
		und = fig_und.add_subplot(8, 7, 7 *i + 1)
		und.imshow(warped)
		und = fig_und.add_subplot(8, 7, 7 * i + 2)
		und.imshow(xFilter, cmap='gray')
		und = fig_und.add_subplot(8, 7, 7 * i + 3)
		und.imshow(yFilter, cmap='gray')
		und = fig_und.add_subplot(8, 7, 7 * i + 4)
		und.imshow(magFilter, cmap='gray')
		und = fig_und.add_subplot(8, 7, 7 * i + 5)
		und.imshow(dirFilter, cmap='gray')
		und = fig_und.add_subplot(8, 7, 7 * i + 6)
		und.imshow(colorFilter, cmap='gray')
		und = fig_und.add_subplot(8, 7, 7 * i + 7)
		und.imshow(combine, cmap='gray')
		finder = FLane.Lane()
		finder.processing(combine, warped,_visualization = True)
		plt.imshow(finder.out_img)
		plt.show()
		finder.processing(combine, warped, _visualization = True)
		plt.imshow(finder.result)
		plt.show()	
		inv_M = cv2.getPerspectiveTransform(dst_p, src_p)
		sign_img = np.ones_like(finder.result) * 255
		inv_warped = cv2.warpPerspective(finder.result, inv_M, (img.shape[1], img.shape[0]))
		inv_sign_img = cv2.warpPerspective(sign_img, inv_M, (img.shape[1], img.shape[0]))
		plt.imshow(inv_warped)
		plt.show()
		
		cover = np.zeros_like(xFilter)
		cover[:, 130:1150] = 1
		plt.imshow(cover, cmap='gray')
		plt.show()
		colorFilter_cp = np.copy(colorFilter)
		colorFilter_cp[(cover == 0)] = 0
		plt.imshow(colorFilter_cp)
		plt.show()
		processed_img = cv2.cvtColor(cv2.imread(test_img_list[-1]), cv2.COLOR_BGR2RGB)
		processed_img[(inv_sign_img == 255)] = inv_warped[(inv_sign_img == 255)]
		plt.imshow(processed_img)
		plt.show()
	# f = plt.show()

	
	




	userhome = os.path.expanduser('~')
	# file_path = userhome+"/Desktop/images"
	file_path = userhome+"/Desktop/Finding-Lane-Lines-on-the-Road/data/test_images"
	# if os.path.exists(file_path):
	# 	print("123")
	# 	files = os.listdir(file_path)
	# 	print(files)
	# 	for file in files:
	# 		print(file)
	# 		image_path = os.path.join(file_path, file)
	# 		# image = cv2.imread(image_path)
	# 		print(image_path)
	# 		image = mpimg.imread(image_path)
	# 		print("This image is:", type(image), "with dim:", image.shape)
	# 		img_copy = np.copy(image)

	# 		rgb_threshold = [200, 200, 40]
	# 		selected_pixel = (image[:, :, 0] < rgb_threshold[0]) | (image[:, :, 1] < rgb_threshold[1]) | (image[:, :, 2] < rgb_threshold[2]) 
			
	# 		lb1 = [145,539]
	# 		rb1 = [910, 539]
	# 		lt1 = [450, 325]
	# 		rt1 = [520, 325]
	# 		roi1 = RegionOfInterest(img_copy, np.array([[lt1, rt1, rb1,lb1]], dtype=np.int32))
	# 		tri_region = (roi1[:, :, 0] > 0) | (roi1[:, :, 1] > 0) | (roi1[:, :, 2] > 0)
	# 		gray = GrayScale(image)
	# 		blur = GaussianBlur(gray, 5)
	# 		canny = Canny(blur, 50, 150) & tri_region
			
	# 		rho = 0.5
	# 		theta = np.pi/180
	# 		threshold = 15

	# 		line_img = HoughLines(canny, rho, theta, threshold, 30, 90, lt1, rt1)
	# 		line = (line_img[:, :, 0] == 255)
			
	# 		img_copy[~selected_pixel & tri_region | line] =[255, 0, 0]

	# 		output = ImageWeighted(img_copy, image, 0.8, 0.2)
	# 		mpimg.imsave('fig/' + 'result.jpg', output)
	# 		mpimg.imsave('fig/' + 'roi.jpg', roi1)
	# 		mpimg.imsave('fig/' + 'color_filiter.jpg', img_copy)
	# 		mpimg.imsave('fig/' + 'canny.jpg', canny, cmap='gray')
	# 		mpimg.imsave('fig/' + 'line.jpg', line_img)
	# 		plt.imshow(output)
	# 		plt.show()
	# 		# assert(0)

			
