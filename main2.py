import cv2
from pupil_detection import PupilDetection

pupil_detection = PupilDetection()

img = cv2.imread('rdg.jpg',0)
center = pupil_detection.detect_pupil(img)
cv2.circle(img, center, 5, (255,0,0))
cv2.imshow('final', img)
cv2.waitKey(0)
