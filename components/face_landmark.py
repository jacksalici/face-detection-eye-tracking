import numpy as np
import dlib
from imutils import face_utils
import cv2

class FaceLandmarkTracking():
    def __init__(self) -> None:
        self.landmark_predictor = dlib.shape_predictor(
            'predictors/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()

    def face_analysis(self, frame):
        faces = self.face_detector(frame,  1)
        poi_list = []
        for face in faces:

            shape = face_utils.shape_to_np(self.landmark_predictor(frame, face))
            (x, y, w, h) = face_utils.rect_to_bb(face)

            poi_list.append({
                "box": (x, y, w, h),
                "traits": {
                    "ear_dx": shape[0],
                    "ear_sx": shape[16],
                    "nose": shape[30],
                    "mouth_dx": shape[48],
                    "mouth_sx": shape[54],
                    "chin": shape[9],
                    "center_eye_dx": np.round(np.mean(shape[36:42], axis=0)).astype(int),
                    "center_eye_sx": np.round(np.mean(shape[42:48], axis=0)).astype(int),

                },
                "eye_corners": {
                    "dx_out": shape[36],
                    "dx_in": shape[39],
                    "sx_out": shape[45],
                    "sx_in": shape[42],
                },
                "eye_edges":{
                    "sx_top": np.round(np.mean(shape[37:38], axis=0)).astype(int),
                    "sx_bottom": np.round(np.mean(shape[41:42], axis=0)).astype(int),
                    "dx_top": np.round(np.mean(shape[43:45], axis=0)).astype(int),
                    "dx_bottom": np.round(np.mean(shape[46:47], axis=0)).astype(int),
                }
            })
            

        return poi_list
