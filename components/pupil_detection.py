import numpy as np
import cv2
from math import sqrt
from queue import *


class PupilDetection():
    def __init__(self, accuracy = 50, blur_size = 5) -> None:
        self.ACCURACY = accuracy
        self.BLUR_SIZE = blur_size
    
    def _unscale_point(self, point, orig):
        h, w = orig.shape
        ratio = self.ACCURACY/w
        return (int(round(point[0]/ratio)),int(round(point[1]/ratio)))
        
    def _scale_to_fast_size(self,src):
        rows, cols = src.shape
        return cv2.resize(src, (self.ACCURACY, int((self.ACCURACY / cols) * rows)))
    
    def _test_possible_centers(self, x, y, gx, gy, arr, weight_matrix, enable_weight=True, weight_divisor=1):
        rows, cols = arr.shape
        for row in range (rows):
            for col in range(cols):
                dx, dy = x-row, y-col 
                
                if dx==0 and dy==0:
                    continue
               
                magnitude = sqrt(dx*dx+dy*dy)
                dx, dy = dx/magnitude, dy/magnitude
                
                dot_product = max(0.0, dx * gx + dy * gy)
                
                arr[row][col] += dot_product * dot_product * int(enable_weight)*(weight_matrix[row][col]/weight_divisor)
        
        return arr
    
    def _get_magnitude_matrix(self, mat_x,mat_y):
        rows, cols, = np.shape(mat_x)
        matrix = np.zeros((rows,cols))
        for row in range(rows):
            for col in range(cols):
                gx = mat_x[row][col]
                gy = mat_y[row][col]
                matrix[row][col]=sqrt(gx*gx+gy*gy)
        return matrix
    
    def _compute_threshold(self, magnitude_matrix, factor=50.0):
        mean_magn_grad, std_magn_grad = cv2.meanStdDev(magnitude_matrix)
        rows, cols = np.shape(magnitude_matrix)
        std_dev = std_magn_grad[0]/sqrt(rows*cols)
        return factor * std_dev + mean_magn_grad[0]

    def _flood_should_push_point(self, dir, mat):
        px, py = dir
        rows, cols = np.shape(mat)
        return px >= 0 and px < cols and py >= 0 and py < rows
        
    def _flood_kill_edges(self, mat):
        rows, cols = np.shape(mat)
        
        cv2.rectangle(mat, (0,0), (cols, rows), 255)
        
        mask = np.ones((rows, cols), dtype=np.uint8)*255
        
        queue = Queue()
        
        queue.put((0,0))
        
        while(queue.qsize()>0):
            px,py=queue.get()
            if mat[py][px]==0:
                continue
            
            for point in [(px+1,py), (px-1,py), (px,py-1), (px,py+1)]:
                if self._flood_should_push_point(point, mat):
                    queue.put(point)
            
            mat[py][px]=0.0
            mask[py][px]=0
    
        return mask

    def _compute_mat_x_gradient(self, mat): 
        rows, cols = mat.shape
        out = np.zeros((rows, cols), dtype='float64')
        mat = mat.astype(float)
        for row in range(rows):
            out[row][0] = mat[row][1] - mat[row][1]
            for col in range(cols - 1):
                out[row][col] = (mat[row][col+1] - mat[row][col-1])/2.0
            out[row][cols - 1] = (mat[row][cols - 1] - mat[row][cols - 2])
        return out
    
    def detect_pupil (self, img):
        rows, cols = np.asarray(img).shape
        
        resized=self._scale_to_fast_size(img)
        
        resized_array=np.asarray(resized)
        
        resized_rows, resized_cols = np.shape(resized_array)
        
        # compute gradient for each point
        grad_arr_x = self._compute_mat_x_gradient(resized_array)
        grad_arr_y = np.transpose(self._compute_mat_x_gradient(np.transpose(resized_array)))

        magnitude_matrix = self._get_magnitude_matrix(grad_arr_x, grad_arr_y)

        # find a threshold, element below that will be put to 0, scaled otherwise
        gradient_threshold = self._compute_threshold(magnitude_matrix)

        for row in range(resized_rows):
            for col in range(resized_cols):
                gx = grad_arr_x[row][col]
                gy = grad_arr_y[row][col]
                mag = magnitude_matrix[row][col]
                grad_arr_x[row][col] = gx/mag if mag>gradient_threshold else 0.0
                grad_arr_y[row][col] = gy/mag if mag>gradient_threshold else 0.0
        
        #create weights
        weight = np.asarray(cv2.GaussianBlur(resized, (self.BLUR_SIZE, self.BLUR_SIZE), 0,0))
        weight_rows, weight_cols = weight.shape
        
        weight = 255 - weight
        
        out_sum = np.zeros((resized_rows, resized_cols))
        out_sum_rows, out_sum_cols = np.shape(out_sum)
        
        # test every possible center
        for row in range (weight_rows):
            for col in range (weight_cols):
                gx, gy = grad_arr_x[row][col], grad_arr_y[row][col]
                if gx==0.0 and gy==0:
                    continue
                self._test_possible_centers(col, row, gx, gy, out_sum, weight)
        
        num_gradients = weight_rows*weight_cols
        out = out_sum.astype(np.float32)*(1/num_gradients)
        
        _, max_val, _, max_p = cv2.minMaxLoc(out)
        #post_processing
        
        flood_threshold = max_val*0.6
        ret, flood_clone = cv2.threshold(out, flood_threshold, 0.0, cv2.THRESH_TOZERO)
        mask = self._flood_kill_edges(flood_clone)
        _, max_val, _, max_p = cv2.minMaxLoc(out, mask)
        
        return self._unscale_point(max_p, img)
        