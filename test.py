# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

from os.path import isfile, join


def convert_frames_to_video(pathIn, pathOut, fps):
	frame_array = []
	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

	# files.sort(key = lambda x: int(x[5:-4]))
	files.sort()
	count = 0
	for i in range(len(files)):
		if "1548211800933442312" in files[i]:
			while count !=3000:
				print(count)
				filename = pathIn + files[i+count]
				img = cv2.imread(filename)
				h, w, l = img.shape
				size = (w,h)
				frame_array.append(img)
				count += 1

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') 
	out = cv2.VideoWriter(pathOut, fourcc, fps, size)

	for i in range(len(frame_array)):
		out.write(frame_array[i])
	out.release()

def main():
	userhome = os.path.expanduser('~')
	file_path = userhome+"/Desktop/rectified_images/"
	pathIn = file_path
	pathOut = userhome+"/Desktop/video.avi"
	fps = 25.0
	convert_frames_to_video(pathIn, pathOut, fps)

if __name__ == "__main__":
	main()