import cv2

BLUR_RADIUS = 5


class PupilDetection():
    def __init__(self) -> None:
        pass
    
    def detect_pupil(self, eye):
        if len(eye.shape)>2:
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur
        blurred_eye = cv2.GaussianBlur(eye, (BLUR_RADIUS, BLUR_RADIUS), 0)
        # Adaptive threshold
        threshold = cv2.adaptiveThreshold(blurred_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
        # Contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Centroid
        M = cv2.moments(largest_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
                    
        return (centroid_x, centroid_y)