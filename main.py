from face_detection_with_haar_cascade import HaarCascade
from gaze_tracking import GazeTracking
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)
cascade = HaarCascade()
gazeTracking = GazeTracking()

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    #cv2.imshow('frame', cascade.eye_framing(cascade.face_detection(frame), frame))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',  gazeTracking.face_analysis(gray))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
