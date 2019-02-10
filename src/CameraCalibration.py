import cv2
import numpy as np
import glob

class Calibration :
    def __init__(self, _dir, _data_format, _nx, _ny) :
        self.dir = _dir
        self.nx = _nx
        self.ny = _ny
        self.format = _data_format

        self.data_list = glob.glob((self.dir + '/*.' + self.format))
        print ("total image : %d"%(len(self.data_list)))

        self.objp = np.zeros((self.nx * self.ny, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

        self.objpoints = []
        self.imgpoints = []

    def getDataList(self) :
        return self.data_list

    def _imgLoader(self, _filename) :
        img = cv2.imread(_filename)
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def findCorner(self, _drawCorner = False) :
        if (_drawCorner) :
            import matplotlib
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(24, 15))
            rows, cols = 5, 4

        for i, _filename in enumerate(self.data_list) :
            if (_drawCorner) :
                img, gray = self._imgLoader(_filename)
                if (i == 0) :
                    self.img_size = (img.shape[1], img.shape[0])
            else :
                _, gray = self._imgLoader(_filename)
                if (i == 0) :
                    self.img_size = (gray.shape[1], gray.shape[0])
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            if (ret) :
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)
                if (_drawCorner) :
                    cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                    ax = fig.add_subplot(rows, cols, i + 1)
                    plt.imshow(img)

        if (_drawCorner) :
            plt.show()

    def cameraCalibration(self, _drawCorner = False) :
        self.findCorner(_drawCorner)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)

    def undistort(self, img) :
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def undistort_list(self, _datalist) : # for test
        import matplotlib
        import matplotlib.pyplot as plt

        fig_ori = plt.figure(figsize=(24, 12))
        fig_ori.suptitle('original image', fontsize=50)

        fig_und = plt.figure(figsize=(24, 12))
        fig_und.suptitle('undistort image', fontsize=50)
        
        rows, cols = 5, 4
        
        for i, _filename in enumerate(_datalist) :
            img, _ = self._imgLoader(_filename)
            und_img = self.undistort(img)

            ori = fig_ori.add_subplot(rows, cols, i + 1)
            ori.imshow(img)
            
            und = fig_und.add_subplot(rows, cols, i + 1)
            und.imshow(und_img)

        plt.show()






