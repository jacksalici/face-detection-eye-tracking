import numpy as np
import dlib

class GazeTracking():
    def __init__(self) -> None:
        self.landmark_predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()
    
    def face_analysis(self, frame):
        faces = self.face_detector(frame,  1)
        for face in faces:
            shape = self.landmark_predictor(frame, face)

            
            
            
    