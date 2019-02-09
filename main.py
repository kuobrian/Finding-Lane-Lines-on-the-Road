# -*- coding: utf-8 -*-
from utils import *
import os
import sys
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
	userhome = os.path.expanduser('~')
	file_path = userhome+"/Desktop/images"
	if os.path.exists(file_path):
		files = os.listdir(file_path)
		f, axarr = plt.subplots(2,2)
		for file in files:
			print(file)
			image_path = os.path.join(file_path, file)
			image = cv2.imread(image_path)
			print("This image is:", type(image), "with dim:", image.shape)
			# mask = get_edges(image)
			# flatten = flatten_perspective(image)
			# lanetracker = LaneTracker(image)
			# lanetracker.process(image)
			# axarr[0,0].imshow(image)
			# axarr[0,1].imshow(mask)
			# axarr[1,0].imshow(flatten)
			# plt.imshow(flatten)
			# plt.show()

			assert(0)
