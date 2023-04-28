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
            
            shape = face_utils.shape_to_np(self.landmark_predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face))
            (x, y, w, h) = face_utils.rect_to_bb(face)
            
            
            landmark_dict = {
                "ear_dx": shape[0],
                "ear_sx": shape[16],
                "nose": shape[30],
                "mouth_dx": shape[48],
                "mouth_sx": shape[54],
                "center_eye_dx": np.round(np.mean(shape[36:42], axis=0)).astype(int),
                "center_eye_sx": np.round(np.mean(shape[42:48], axis=0)).astype(int),
            }
                        
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            for mark in landmark_dict.values():
                cv2.circle(frame, (mark[0], mark[1]), 2, (0, 255, 255), -1)
            
        return frame
            
    