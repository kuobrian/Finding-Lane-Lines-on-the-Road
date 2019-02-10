import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Lane :
    def __init__(self, _nwindows = 9, _margin = 100, _minpix = 50, _n_iter = 30) :
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        # Set the width of the windows +/- margin
        # Set minimum number of pixels found to recenter window
        self.nwindows = _nwindows
        self.margin = _margin
        self.minpix = _minpix

        self.Line_thinkness = 15

        self.n_iter = _n_iter

        self.ploty = None

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

        self.out_img = []
        self.histogram = []
        self.result = []

        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        # self.recent_xfitted = [] 
        self.recent_xfitted_left = []
        self.recent_xfitted_right = []
        #average x values of the fitted line over the last n iterations
        self.bestx_left = None  
        self.bestx_right = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_left = None
        self.best_fit_right = None
        #polynomial coefficients for the most recent fit
        self.current_fit_left = [np.array([False])]  
        self.current_fit_right = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        self.diff_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs_left = np.array([0,0,0], dtype='float') 
        self.diffs_right = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        # self.allx = None  
        #y values for detected line pixels
        # self.ally = None  

    def find_lane_pixels(self, _img_b, _draw_img, _visualization = False) :
        # Take a histogram of the bottom half of the image
        histogram = np.sum(_img_b[_img_b.shape[0]//2:,:], axis=0)
        self.histogram = np.copy(histogram)
        # Create an output image to draw on and visualize the result
        if (_visualization) :
            out_img = np.dstack((_img_b, _img_b, _img_b))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on self.nwindows above and image shape
        window_height = np.int(_img_b.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = _img_b.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in self.nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = _img_b.shape[0] - (window+1)*window_height
            win_y_high = _img_b.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - self.margin  # Update this
            win_xleft_high = leftx_current + self.margin  # Update this
            win_xright_low = rightx_current - self.margin  # Update this
            win_xright_high = rightx_current + self.margin  # Update this
            
            # Draw the windows on the visualization image
            if (_visualization) :
                cv2.rectangle(_draw_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(_draw_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### TO-DO: If you found > self.minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            # pass # Remove this when you add your function
            if (len(good_left_inds) > self.minpix) :
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if (_visualization) :
            self.out_img = np.copy(_draw_img)

        return leftx, lefty, rightx, righty


    def fit_polynomial(self, _img_b, _leftx, _lefty, 
                            _rightx, _righty, 
                            _visualization = False) :
        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        left_fit = np.polyfit(_lefty, _leftx, 2)
        right_fit = np.polyfit(_righty, _rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, _img_b.shape[0]-1, _img_b.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # if (_visualization) :
        #     # Colors in the left and right lane regions
        #     self.out_img[_lefty, _leftx] = [255, 0, 0]
        #     self.out_img[_righty, _rightx] = [0, 0, 255]

        #     # Plots the left and right polynomials on the lane lines
        #     plt.plot(left_fitx, ploty, color='yellow')
        #     plt.plot(right_fitx, ploty, color='yellow')
        
        return left_fitx, right_fitx, left_fit, right_fit

    def search_around_poly(self, _img_b, _draw_img, _left_fit, _right_fit, 
                            _visualization = False) :
        # Grab activated pixels
        nonzero = _img_b.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = (nonzerox > (_left_fit[0] * nonzeroy ** 2 + _left_fit[1] * nonzeroy + _left_fit[2] - self.margin)) & \
                        (nonzerox < (_left_fit[0] * nonzeroy ** 2 + _left_fit[1] * nonzeroy + _left_fit[2] + self.margin))
        right_lane_inds = (nonzerox > (_right_fit[0] * nonzeroy ** 2 + _right_fit[1] * nonzeroy + _right_fit[2] - self.margin)) & \
                        (nonzerox < (_right_fit[0] * nonzeroy ** 2 + _right_fit[1] * nonzeroy + _right_fit[2] + self.margin))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, left_fit, right_fit = self.fit_polynomial(_img_b, leftx, lefty, 
                                    rightx, righty, _visualization = _visualization)
        ploty = np.linspace(0, _img_b.shape[0]-1, _img_b.shape[0])

        ## Visualization ##
        # Show search margin
        '''
        if _visualization :
            # Create an image to draw on and an image to show the selection window
            # out_img = np.dstack((_img_b, _img_b, _img_b))*255
            window_img = np.zeros_like(_draw_img)
            # Color in left and right line pixels
            _draw_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            _draw_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, 
                                    ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, 
                                    ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(_draw_img, 1, window_img, 0.3, 0)
            
            # Plot the polynomial lines onto the image
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            ## End visualization steps ##
            
            self.result = np.copy(result)
        '''
        if _visualization :
            # Create an image to draw on and an image to show the selection window
            # out_img = np.dstack((_img_b, _img_b, _img_b))*255
            window_img = np.zeros_like(_draw_img)
            # Color in left and right line pixels
            # _draw_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            # _draw_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.Line_thinkness, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.Line_thinkness, 
                                    ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.Line_thinkness, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.Line_thinkness, 
                                    ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            center_area_pts = np.hstack((left_line_window2, right_line_window1))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 0, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))
            cv2.fillPoly(window_img, np.int_([center_area_pts]), (0, 255, 0))

            result = cv2.addWeighted(_draw_img, 1, window_img, 0.4, 0)
            
            # Plot the polynomial lines onto the image
            # plt.plot(left_fitx, ploty, color='yellow')
            # plt.plot(right_fitx, ploty, color='yellow')
            ## End visualization steps ##
            
            self.result = np.copy(result)

        return left_fitx, right_fitx, left_fit, right_fit

    def measure_curvature_real(self, _img_b, _left_fitx, _right_fitx):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        ploty = np.linspace(0, _img_b.shape[0]-1, _img_b.shape[0])
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, _left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, _right_fitx * self.xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        
        y_eval = np.max(ploty)

        ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix+ left_fit_cr[1])**2 )**1.5) / (2 * np.abs(left_fit_cr[0]))
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1])**2 )**1.5) / (2 * np.abs(right_fit_cr[0]))

        return left_curverad, right_curverad

    def processing(self, _img_b, _draw_img, _visualization = False) :
        if (self.detected) :
            left_fitx, right_fitx, left_fit, right_fit = self.search_around_poly(_img_b, _draw_img, self.current_fit_left, self.current_fit_right
                                , _visualization = _visualization)
            left_curverad, right_curverad = self.measure_curvature_real(_img_b, left_fitx, right_fitx)
            
            self.recent_xfitted_left.append(left_fitx)
            self.recent_xfitted_right.append(right_fitx)

            self.diffs_left = self.current_fit_left - left_fit
            self.diffs_right = self.current_fit_right - right_fit

            self.current_fit_left = left_fit 
            self.current_fit_right = right_fit  

            self.diff_pos = ((right_curverad - left_curverad) / 2) + left_curverad - self.line_base_pos

            self.line_base_pos = ((right_curverad - left_curverad) / 2) + left_curverad
            self.radius_of_curvature = (left_curverad, right_curverad) 

            if (len(self.recent_xfitted_left) > self.n_iter) :
                self.recent_xfitted_left.remove(self.recent_xfitted_left[0])
                self.recent_xfitted_right.remove(self.recent_xfitted_right[0])
            
            self.getBest()

        else :
            self.ploty = np.linspace(0, _img_b.shape[0]-1, _img_b.shape[0])
            leftx, lefty, rightx, righty = self.find_lane_pixels(_img_b, _draw_img, _visualization = _visualization)
            left_fitx, right_fitx, left_fit, right_fit = self.fit_polynomial(_img_b, leftx, lefty, 
                                                            rightx, righty, _visualization=_visualization)
            left_curverad, right_curverad = self.measure_curvature_real(_img_b, left_fitx, right_fitx)
            
            self.detected = True
            
            self.recent_xfitted_left.append(left_fitx)
            self.recent_xfitted_right.append(right_fitx)

            self.current_fit_left = left_fit 
            self.current_fit_right = right_fit 

            self.line_base_pos = ((right_curverad - left_curverad) / 2) + left_curverad
            self.radius_of_curvature = (left_curverad, right_curverad) 
            left_fitx, right_fitx, left_fit, right_fit = self.search_around_poly(_img_b, _draw_img, self.current_fit_left, self.current_fit_right
                                , _visualization = _visualization)

    def getBest(self) :
        self.bestx_left = np.mean(self.recent_xfitted_left, 0)
        self.bestx_right =  np.mean(self.recent_xfitted_right, 0)

        self.best_fit_left = np.polyfit(self.ploty, self.bestx_left, 2)
        self.best_fit_right = np.polyfit(self.ploty, self.bestx_right, 2)

        # left_fitx point 
        # left_fit coeff

        # was the line detected in the last iteration?
        # self.detected = False  
        # x values of the last n fits of the line
        # self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        # self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        # self.best_fit = None  
        #polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        # self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        # self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        # self.allx = None  
        #y values for detected line pixels
        # self.ally = None 
