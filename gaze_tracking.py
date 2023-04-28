import numpy as np
import dlib
from imutils import face_utils
import cv2

class GazeTracking():
    def __init__(self) -> None:
        self.landmark_predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()
    
    def face_analysis(self, frame):
        faces = self.face_detector(frame,  1)
        for face in faces:
            shape = self.landmark_predictor(frame, face)

  
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
        return frame
            
    