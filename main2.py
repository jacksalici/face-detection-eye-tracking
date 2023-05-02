
from face_detection_with_haar_cascade import HaarCascade
from face_landmark import FaceLandmarkTracking
from pupil_detection import PupilDetection
import cv2


landmark_tracking = FaceLandmarkTracking()

pupil_detection = PupilDetection()

frame = cv2.imread("rdg.jpg")
framebg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for face in landmark_tracking.face_analysis(framebg):
    
    for mark in face["eye_corners"].values():
        cv2.circle(frame, (mark[0], mark[1]), 2, (255, 0, 255), -1)
    
    
    for side in ["dx", "sx"]:
        cy = face["traits"]["center_eye_"+side][0]
        cx = face["traits"]["center_eye_"+side][1]
        
        
        w = int((abs(face["eye_corners"][side+"_in"][1] - face["eye_corners"][side+"_out"][1]) + 100 )/2)
            
        cv2.rectangle(frame, (cy-w, cx-w), (cy+w, cx+w), (0, 255, 255), 2)
        
        new_img_bg = framebg[cx-w:cx+w, cy-w:cy+w]
        ncy, ncx = pupil_detection.detect_pupil(framebg[cx-w:cx+w, cy-w:cy+w])
        
        ncy, ncx = ncy+cy-w, ncx+cx-w
        cv2.circle(frame, (ncy, ncx), 2, (255, 255, 0), -1)

cv2.imshow('frame',  frame)
cv2.waitKey(10000)
cv2.imwrite('rdg_detected.jpg', frame)

