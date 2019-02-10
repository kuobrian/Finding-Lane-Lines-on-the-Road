import cv2
import numpy as np

Half_PI = np.pi / 2

class gradFilters :
    def __init__(self, _sobel_x_thresh = (0, 255), 
                       _sobel_y_thresh = (0, 255), 
                       _mag_thresh = (0, 255), 
                       _dir_thresh = (0, Half_PI)
                       ) :
        self.sobel_x_thresh = _sobel_x_thresh
        self.sobel_y_thresh = _sobel_y_thresh
        self.mag_thresh = _mag_thresh
        self.dir_thresh = _dir_thresh

        self.x_grad = None
        self.y_grad = None
        self.mag_grad = None
        self.dir = None

        self.scaled_x_grad = None
        self.scaled_y_grad = None
        self.scaled_mag_grad = None

        self.x_grad_filter = None
        self.y_grad_filter = None
        self.mag_filter = None
        self.dir_filter = None

    def compute_x_grad(self, _img, _kernel) :
        if (len(_img.shape) == 3) :
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        x_grad = cv2.Sobel(_img, cv2.CV_64F, 1, 0, ksize=_kernel)
        x_grad = np.abs(x_grad)
        scaled_x_grad = np.uint8(255 * x_grad / np.max(x_grad))
        return x_grad, scaled_x_grad

    def compute_y_grad(self, _img, _kernel) :
        if (len(_img.shape) == 3) :
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        y_grad = cv2.Sobel(_img, cv2.CV_64F, 0, 1, ksize=_kernel)
        y_grad = np.abs(y_grad)
        scaled_y_grad = np.uint8(255 * y_grad / np.max(y_grad))
        return y_grad, scaled_y_grad

    def compute_mag_grad(self, _img, _kernel) :
        # if (self.x_grad == None) :
        self.x_grad, self.scaled_x_grad = self.compute_x_grad(_img, _kernel)
        # if (self.y_grad == None) :
        self.y_grad, self.scaled_y_grad = self.compute_y_grad(_img, _kernel)

        self.mag_grad = np.abs((self.x_grad ** 2 + self.y_grad ** 2) ** 0.5)
        self.scaled_mag_grad = np.uint8(255 * self.mag_grad / np.max(self.mag_grad))
        return self.mag_grad, self.scaled_mag_grad

    def compute_dir(self, _img, _kernel) :
        # if (self.x_grad == None) :
        x_grad, _ = self.compute_x_grad(_img, _kernel)
        # if (self.y_grad == None) :
        y_grad, _ = self.compute_y_grad(_img, _kernel)

        self.dir = np.arctan2(y_grad, x_grad)
        return self.dir

    def getXgradFilter(self, _thresh = (0, 255)) :
        if ((_thresh != (0, 255)) and (_thresh != self.sobel_x_thresh)) :
            self.sobel_x_thresh = _thresh
        self.x_grad_filter = np.zeros_like(self.scaled_x_grad)
        self.x_grad_filter[(self.scaled_x_grad >= self.sobel_x_thresh[0]) &  
                            (self.scaled_x_grad <= self.sobel_x_thresh[1])] = 1
        return self.x_grad_filter

    def getYgradFilter(self, _thresh = (0, 255)) :
        if ((_thresh != (0, 255)) and (_thresh != self.sobel_y_thresh)) :
            self.sobel_y_thresh = _thresh
        self.y_grad_filter = np.zeros_like(self.scaled_y_grad)
        self.y_grad_filter[(self.scaled_y_grad >= self.sobel_y_thresh[0]) &  
                            (self.scaled_y_grad <= self.sobel_y_thresh[1])] = 1
        return self.y_grad_filter    

    def getMagFilter(self, _thresh = (0, 255)) :
        if ((_thresh != (0, 255)) and (_thresh != self.mag_thresh)) :
            self.mag_thresh = _thresh
        self.mag_filter = np.zeros_like(self.scaled_mag_grad)
        self.mag_filter[(self.scaled_mag_grad >= self.mag_thresh[0]) &  
                            (self.scaled_mag_grad <= self.mag_thresh[1])] = 1
        return self.mag_filter

    def getDirFilter(self, _thresh = (0, Half_PI)) :
        if ((_thresh != (0, Half_PI)) and (_thresh != self.dir_thresh)) :
            self.dir_thresh = _thresh
        self.dir_filter = np.zeros_like(self.dir)
        self.dir_filter[(self.dir >= self.dir_thresh[0]) &  
                        (self.dir <= self.dir_thresh[1])] = 1
        return self.dir_filter

    # def getFusing(self, _X_thresh = (0, 255), 
    #                     _Y_thresh = (0, 255), 
    #                     _Mag_thresh = (0, 255), 
    #                     _Dir_thresh = (0, Half_PI)) :
    #     if ((_X_thresh != (0, 255)) and (_X_thresh != self.sobel_x_thresh)) :
    #         self.getXgradFilter(_X_thresh)
    #     if ((_Y_thresh != (0, 255)) and (_Y_thresh != self.sobel_y_thresh)) :
    #         self.getYgradFilter(_Y_thresh)
    #     if ((_Mag_thresh != (0, 255)) and (_Mag_thresh != self.mag_thresh)) :
    #         self.getMagFilter(_Mag_thresh)
    #     if ((_Dir_thresh != (0, Half_PI)) and (_Dir_thresh != self.dir_thresh)) :
    #         self.getDirFilter(_Dir_thresh)

class colorFilter :
    def __init__(self, _color_space = 'HLS', _s_thresh = (0, 255)) :
        if (_color_space == 'HLS') :
            self.color_space = cv2.COLOR_BGR2HLS
        self.s_thresh = _s_thresh

        self.converted = False
        self.converted_img = None
        self.color_filter = None

    def getColorFilter(self, _img, _thresh = (0, 255)) :
        self.converted_img = cv2.cvtColor(_img, self.color_space)
        if ((_thresh != (0, 255)) and (_thresh != self.s_thresh)) :
            self.s_thresh = _thresh

        self.color_filter = np.zeros_like(self.converted_img[:, :, 2])
        self.color_filter[(self.converted_img[:, :, 2] > self.s_thresh[0]) &  
                            (self.converted_img[:, :, 2] <= self.s_thresh[1])] = 1
        return self.color_filter