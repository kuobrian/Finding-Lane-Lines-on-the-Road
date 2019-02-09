import math

def GrayScale(image):
	return cv2.cvtoColor(image, cv2.COLOR_RGB2GRAY)

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
