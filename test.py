import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt

import src.CameraCalibration as cc
import src.Filters as F

def CCalibration(img) :
    calibration = cc.Calibration('./camera_cal', 'jpg', 9, 6)
    # print (calibration.getDataList())
    # _list = calibration.getDataList()
    # calibration.findCorner(_drawCorner = True)
    # calibration.findCorner()
    calibration.cameraCalibration()
    # calibration.cameraCalibration(_drawCorner = True)
    # calibration.undistort_list(_list)
    return calibration.undistort(img)

def Cfilter() :
    img_list = glob.glob('./colorspace_test_images/*.jpg')
    img = cv2.imread(img_list[1])
    colorF = F.colorFilter()
    img = colorF.getColorFilter(img, (100, 255))
    plt.imshow(img, cmap = 'gray')
    plt.show()

def Gfilter() :
    img_list = glob.glob('./colorspace_test_images/*.jpg')
    img = cv2.imread(img_list[1])
    gradF = F.gradFilters(_sobel_x_thresh=(20, 100), 
                            _sobel_y_thresh=(20, 100), 
                            _mag_thresh=(20, 100), 
                            _dir_thresh=(0.7, 1.3))

    _, _ = gradF.compute_mag_grad(img, 3)
    _, _ = gradF.compute_dir(img, 15)

    xFilter = gradF.getXgradFilter()
    yFilter = gradF.getYgradFilter()
    magFilter = gradF.getMagFilter()
    dirFilter = gradF.getDirFilter()

    

if "__main__" :
    # CCalibration()
    # Cfilter()
    test_img_list = glob.glob('./test_images/*.jpg')
    plt.imshow(CCalibration(cv2.cvtColor(cv2.imread(test_img_list[0]), cv2.COLOR_BGR2RGB)))
    plt.show()