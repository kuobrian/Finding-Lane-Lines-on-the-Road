import math
import cv2 as cv2
import numpy as np

def GrayScale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def Canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def GaussianBlur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def RegionOfInterest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def HoughLines(img, rho, theta, threshold, min_line_len, max_line_gap, l_t, r_t):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    DrawLines(line_img, lines, l_t, r_t)
    return line_img

def DrawLines(img, lines, l_t, r_t, color=[255, 0, 0] ,thickness=8):
    plus_slope, minus_slope, plus_i, minus_i = ComputeSlope(lines)
    _m_p = np.sort(np.array(plus_slope)[:, 1])[(len(plus_slope) // 2)]
    indices_p = np.where(np.array(plus_slope)[:, 1] == _m_p)
    _m_p = np.mean(np.array(plus_slope)[:, 1])
    
    __b_l = int((img.shape[0] - 1 - lines[ plus_slope[ indices_p[0][0] ][0] ][0][1] + _m_p * lines[ plus_slope[ indices_p[0][0] ][0] ][0][0]) / _m_p)
    __t_l = int((img.shape[0] - 1 - l_t[1] - _m_p * __b_l) / -_m_p)
                
    _m_m = np.sort(np.array(minus_slope)[:, 1])[(len(minus_slope) // 2)]
    indices_m = np.where(np.array(minus_slope)[:, 1] == _m_m)
    _m_m = np.mean(np.array(minus_slope)[:, 1])

    __b_r = int((img.shape[0] - 1 - lines[ minus_slope[ indices_m[0][0] ][0] ][0][3] + _m_m * lines[ minus_slope[ indices_m[0][0] ][0] ][0][2]) / _m_m)
    __t_r = int((img.shape[0] - 1 - r_t[1]  - _m_m * __b_r) / -_m_m)
   
    cv2.line(img, (__t_l, l_t[1]), (__b_l, img.shape[0] - 1), color, thickness)
    cv2.line(img, (__b_r, img.shape[0] - 1), (__t_r, r_t[1]), color, thickness)

def ComputeSlope(lines) :
    plus_slope = []
    minus_slope = []
    plus_i_c, minus_i_c = 0, 0
    for i, line in enumerate(lines) :
        _s = 0
        for x1,y1,x2,y2 in line :
            _s = (y2 - y1) / (x2 - x1)
            if _s > 0 :
                if len(plus_slope) == 0 :
                    plus_i = (i, plus_i_c)
                else :
                    if ((lines[plus_i[0]][0][3] - 539) < (y2 - 539)) :
                        plus_i = (i, plus_i_c)
                plus_slope.append((i, _s))
                plus_i_c += 1
            if _s < 0  :
                if len(minus_slope) == 0 :
                    minus_i = (i, minus_i_c)
                else :
                    if ((lines[minus_i[0]][0][1] - 539) < (y1 - 539)) :
                        minus_i = (i, minus_i_c)
                minus_slope.append((i, _s))
                minus_i_c += 1
        
    
    return plus_slope, minus_slope, plus_i, minus_i

def ImageWeighted(img, initial_img, a=0.8, b=1., c=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, c)