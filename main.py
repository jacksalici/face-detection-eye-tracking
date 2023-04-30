from face_detection_with_haar_cascade import HaarCascade
from face_landmark import FaceLandmarkTracking
from pupil_detection import PupilDetection
import cv2


#vid = cv2.VideoCapture(0)
landmark_tracking = FaceLandmarkTracking()
pupil_detection = PupilDetection()

#while(True):
    #ret, frame = vid.read()
frame = cv2.imread("rdg.jpg")
framebg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for face in landmark_tracking.face_analysis(framebg):
    cy = face["traits"]["center_eye_sx"][0]
    cx = face["traits"]["center_eye_sx"][1]
    
    
    w = int((abs(face["eye_corners"]["sx_in"][1] - face["eye_corners"]["sx_out"][1]) + 100 )/2)
        
    cv2.circle(frame, (cy, cx), w*2, (0, 255, 255), 2)
    
    new_img_bg = framebg[cx-w:cx+w, cy-w:cy+w]

    ncy, ncx = pupil_detection.detect_pupil(framebg[cx-w:cx+w, cy-w:cy+w])
    
    ncy, ncx = ncy+cy-w, ncx+cx-w
    cv2.circle(frame, (ncy, ncx), 2, (0, 255, 255), -1)


cv2.imshow('frame',  frame)
cv2.waitKey(10000)

#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break


#vid.release()
