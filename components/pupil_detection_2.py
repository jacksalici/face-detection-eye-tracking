import cv2
import numpy as np
import random

BLUR_RADIUS = 5


class PupilDetection():
    def __init__(self) -> None:
        pass
    
    def detect_pupil(self, eye, index=random.randint(1,100)):
        if len(eye.shape)>2:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        
        eye = np.clip(eye * 1.25, 0, 255)
        eye = eye.astype('uint8')
        
        eye = cv2.equalizeHist(eye)
        cv2.imshow("h"+str(index), eye)

        # Gaussian blur
        blurred_eye = cv2.GaussianBlur(eye, (BLUR_RADIUS, BLUR_RADIUS), 0)
        
        
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(blurred_eye,kernel,iterations = 1)
        cv2.imshow("e"+str(index), erosion)


        threshold = cv2.adaptiveThreshold(erosion,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,-5)
        
        cv2.imshow("c"+str(index), threshold)
        # Contours
        
        
        # Centroid
        try:
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        except:
            print('error finding moments')
            centroid_x = None
            centroid_y = None
        return (centroid_x, centroid_y)