from face_detection_with_haar_cascade import HaarCascade
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)
cascade = HaarCascade()

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    cv2.imshow('frame', cascade.eye_framing(cascade.face_detection(frame), frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
