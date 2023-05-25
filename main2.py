
from components.face_detection import HaarCascade
from components.face_landmark import FaceLandmarkTracking
from components.pupil_detection import PupilDetection
from components.pnp_solver import PnPSolver
import cv2
import numpy as np


landmark_tracking = FaceLandmarkTracking()
pupil_detection = PupilDetection()
pnp_solver = PnPSolver()

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
        eye_centers = pupil_detection.detect_pupil(framebg[cx-w:cx+w, cy-w:cy+w])
        
        #ncy, ncx = ncy+cy-w, ncx+cx-w
        cv2.circle(frame, eye_centers, 2, (255, 255, 0), -1)
        
        viewPoint=pnp_solver.getCoordFromFace(np.array(list(face["traits"].values()), dtype=np.float32),eye_centers)
        cv2.line(frame,(int(eye_centers[0]),int(eye_centers[1])),(int(eye_centers[0]-viewPoint[0]),int(eye_centers[1]-viewPoint[1])),(0,255,0),4, -1)
 
        
        
    
    

cv2.imshow('frame',  frame)
cv2.waitKey(10000)
cv2.imwrite('rdg_detected.jpg', frame)

