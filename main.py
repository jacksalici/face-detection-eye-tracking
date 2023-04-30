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
cv2.imshow('frame',  frame)
cv2.waitKey(10000)
for face in landmark_tracking.face_analysis(frame):
    for mark in face["eye_corners"].values():
            cv2.circle(frame, (mark[0], mark[1]), 2, (0, 255, 255), -1)
            print(mark)
    
    cv2.imshow('frame',  frame)
    cv2.waitKey(10000)
    
    x = face["eye_corners"]["sx_out"][1]-25
    y = face["eye_corners"]["sx_top"][0]-25
    w = abs(face["eye_corners"]["sx_in"][1] - face["eye_corners"]["sx_out"][1])+50
    h = abs(face["eye_corners"]["sx_bottom"][0] - face["eye_corners"]["sx_top"][0])+50
    print(x,y,w,h)
        
    #cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

cv2.imshow('frame',  frame[y:y+h, x:x+w])
cv2.waitKey(10000)

#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break


#vid.release()
