import numpy as np
import cv2
from math import sqrt
from queue import *

WEIGHT_DIVISOR = 1.0
GRADIENT_THRESHOLD = 20.0
THRESHOLD_VALUE = 0.6
max_eye_size=10

class PupilDetection():
    def __init__(self, accuracy = 40, blur_size = 5) -> None:
        self.ACCURACY = accuracy
        self.BLUR_SIZE = blur_size
    
    def _unscale_point(self, point, orig):
        h, w = orig.shape
        ratio = self.ACCURACY/w
        return (int(round(point[0]/ratio)),int(round(point[1]/ratio)))
        
    def _scale_to_fast_size(self,src):
        rows, cols = src.shape
        return cv2.resize(src, (self.ACCURACY, int((self.ACCURACY / cols) * rows)), interpolation = cv2.INTER_AREA)
    
    def _test_possible_centers(self, x, y, gx, gy, out, weight):
        rows, cols = out.shape
        for row in range (rows):
            for col in range(cols):
                dx, dy = x-row, y-col 
                
                if dx==0 and dy==0:
                    continue
               
                magnitude = sqrt((dx*dx)+(dy*dy))
                dx = dx/magnitude
                dy = dy/magnitude
                
                dot_product = max(0.0, dx * gx + dy * gy)
                
                out[row][col] += dot_product * dot_product * weight[row][col]
    
    def _get_magnitude_matrix(self, mat_x,mat_y):
        rows, cols, = np.shape(mat_x)
        matrix = np.zeros((rows,cols), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                gx = mat_x[row][col]
                gy = mat_y[row][col]
                matrix[row][col]=sqrt((gx*gx)+(gy*gy))
        return matrix
    
    def _compute_dynamic_threshold(self, magnitude_matrix, factor=GRADIENT_THRESHOLD):
        (meanMagnGrad, meanMagnGrad) = cv2.meanStdDev(magnitude_matrix)
        stdDev=meanMagnGrad[0]/sqrt(magnitude_matrix.shape[0]*magnitude_matrix.shape[1])
        return factor*stdDev+meanMagnGrad[0]

    def _flood_should_push_point(self, dir, mat):
        px, py = dir
        rows, cols = np.shape(mat)
        return (px >= 0 and px < cols and py >= 0 and py < rows)
        
    def _flood_kill_edges(self, mat):
        rows, cols = np.shape(mat)
        
        cv2.rectangle(mat, (0,0), (cols, rows), 255)
        
        mask = np.ones((rows, cols), dtype=np.uint8)
        mask = mask*255
        
        queue = Queue()
        
        queue.put((0,0))
        
        while(queue.qsize()>0):
            px,py=queue.get()
            if mat[py][px]==0:
                continue
            
            for point in [(px+1,py), (px-1,py), (px,py+1), (px,py-1)]:
                if self._flood_should_push_point(point, mat):
                    queue.put(point)
            
            mat[py][px]=0.0
            mask[py][px]=0
    
        return mask

    def _compute_gradient(self, img): 
        out = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32) 
        if img.shape[0] < 2 or img.shape[1] < 2: 
            print("EYES too small")
            return out
        for y in range(0,out.shape[0]):
            out[y][0]=img[y][1]-img[y][0]
            for x in range(1,out.shape[1]-1):
                out[y][x]=(img[y][x+1]-img[y][x-1])/2.0
            out[y][out.shape[1]-1]=img[y][out.shape[1]-1]-img[y][out.shape[1]-2]
        return out
    
    def detect_pupil (self, eye_image):
        if (len(eye_image.shape) <= 0 or eye_image.shape[0] <= 0 or eye_image.shape[1] <= 0):
            return (0, 0)
        eye_image = eye_image.astype(np.float32)
        scale_value=1.0
        if(eye_image.shape[0] > max_eye_size or eye_image.shape[1] > max_eye_size):
            scale_value=max(max_eye_size/float(eye_image.shape[0]),max_eye_size/float(eye_image.shape[1]))
            eye_image=cv2.resize(eye_image,None, fx=scale_value,fy= scale_value, interpolation = cv2.INTER_AREA)

        
        # compute gradient for each point
        grad_arr_x = self._compute_gradient(eye_image)
        grad_arr_y = np.transpose(self._compute_gradient(np.transpose(eye_image)))

        magnitude_matrix = self._get_magnitude_matrix(grad_arr_x, grad_arr_y)
        # find a threshold, element below that will be put to 0, scaled otherwise
        gradient_threshold = self._compute_dynamic_threshold(magnitude_matrix)

        for y in range(eye_image.shape[0]):
            for x in range(eye_image.shape[0]):
                if(magnitude_matrix[y][x]>gradient_threshold):
                    grad_arr_x[y][x]=grad_arr_x[y][x]/magnitude_matrix[y][x]
                    grad_arr_y[y][x]=grad_arr_y[y][x]/magnitude_matrix[y][x]
                else:
                    grad_arr_x[y][x]=0.0
                    grad_arr_y[y][x]=0.0
                        
        #create weights
        weight = cv2.GaussianBlur(eye_image, (self.BLUR_SIZE, self.BLUR_SIZE), 0)
        
        weight_rows, weight_cols = np.shape(weight)
        # invert the weight matrix
        for y in range(weight_rows):
            for x in range(weight_cols):
                weight[y][x] = 255-weight[y][x]
        
        out_sum = np.zeros((eye_image.shape[0],eye_image.shape[1]), dtype=np.float32)
        out_sum_rows, out_sum_cols = np.shape(out_sum)
        
        # test every possible center
        for row in range (out_sum_rows):
            for col in range (out_sum_cols):
                gx = grad_arr_x[row][col]
                gy = grad_arr_y[row][col]
                if gx==0.0 and gy==0:
                    continue
                self._test_possible_centers(col, row, gx, gy, out_sum, weight)
        
        num_gradients = weight_rows*weight_cols
        out= np.divide(out_sum, num_gradients*10)
                
        _, max_val, _, max_p = cv2.minMaxLoc(out)
        #post_processing
        
        #flood_threshold = max_val*THRESHOLD_VALUE
        #ret, flood_clone = cv2.threshold(out, flood_threshold, 0.0, cv2.THRESH_TOZERO)
        #mask = self._flood_kill_edges(flood_clone)
        #_, max_val, _, max_p = cv2.minMaxLoc(out, mask)
        max_p=(int(max_p[0]/scale_value),int(max_p[1]/scale_value))
        return max_p
        