import numpy as np
import dlib
from imutils import face_utils
import cv2


class FaceLandmarkTracking():
    def __init__(self) -> None:
        self.landmark_predictor = dlib.shape_predictor(
            'src/resources/predictors/shape_predictor_68_face_landmarks.dat')
        self.face_detector = dlib.get_frontal_face_detector()

    def face_analysis(self, frame):
        faces = self.face_detector(frame,  1)
        poi = []
        for face in faces:

            points = face_utils.shape_to_np(
                self.landmark_predictor(frame, face))
            (x, y, w, h) = face_utils.rect_to_bb(face)

            poi.append({
                "box": (x, y, w, h),
                "ear_dx": points[0],
                "ear_sx": points[16],
                "nose": points[30],
                "mouth_dx": points[48],
                "mouth_sx": points[54],
                "chin": points[8],
                "eye_dx_center": np.round(np.mean(points[36:42], axis=0)).astype(int),
                "eye_sx_center": np.round(np.mean(points[42:48], axis=0)).astype(int),
                "eye_dx_out": points[36],
                "eye_dx_in": points[39],
                "eye_sx_out": points[45],
                "eye_sx_in": points[42],
                "eye_dx_top": np.round(np.mean(points[37:39], axis=0)).astype(int),
                "eye_dx_bottom": np.round(np.mean(points[40:42], axis=0)).astype(int),
                "eye_sx_top": np.round(np.mean(points[43:45], axis=0)).astype(int),
                "eye_sx_bottom": np.round(np.mean(points[46:48], axis=0)).astype(int),
            })


        return poi
