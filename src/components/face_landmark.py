import numpy as np
import dlib
from imutils import face_utils
import cv2


class FaceLandmarkK():
    BOX = 'box'
    EAR_R = 'ear_r'
    EAR_L = 'ear_l'
    NOSE = 'nose'
    MOUTH_R = 'mouth_r'
    MOUTH_L = 'mouth_l'
    CHIN = 'chin'
    EYE_R_CENTER = 'eye_r_center'
    EYE_L_CENTER = 'eye_l_center'
    EYE_R_OUT = 'eye_r_out'
    EYE_R_IN = 'eye_r_in'
    EYE_L_OUT = 'eye_l_out'
    EYE_L_IN = 'eye_l_in'
    EYE_R_TOP = 'eye_r_top'
    EYE_R_BOTTOM = 'eye_r_bottom'
    EYE_L_TOP = 'eye_l_top'
    EYE_L_BOTTOM = 'eye_l_bottom'


class FaceLandmarkTracking():
    def __init__(self, shape_predictor_path) -> None:
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.k = FaceLandmarkK()

    def face_analysis(self, frame):
        faces = self.face_detector(frame,  1)
        poi = []
        for face in faces:

            points = face_utils.shape_to_np(
                self.landmark_predictor(frame, face))
            (x, y, w, h) = face_utils.rect_to_bb(face)

            poi.append({
                self.k.BOX: (x, y, w, h),
                self.k.EAR_R: points[0],
                self.k.EAR_L: points[16],
                self.k.NOSE: points[30],
                self.k.MOUTH_R: points[48],
                self.k.MOUTH_L: points[54],
                self.k.CHIN: points[8],
                self.k.EYE_R_CENTER: np.round(np.mean(points[36:42], axis=0)).astype(int),
                self.k.EYE_L_CENTER: np.round(np.mean(points[42:48], axis=0)).astype(int),
                self.k.EYE_R_OUT: points[36],
                self.k.EYE_R_IN: points[39],
                self.k.EYE_L_OUT: points[45],
                self.k.EYE_L_IN: points[42],
                self.k.EYE_R_TOP: np.round(np.mean(points[37:39], axis=0)).astype(int),
                self.k.EYE_R_BOTTOM: np.round(np.mean(points[40:42], axis=0)).astype(int),
                self.k.EYE_L_TOP: np.round(np.mean(points[43:45], axis=0)).astype(int),
                self.k.EYE_L_BOTTOM: np.round(np.mean(points[46:48], axis=0)).astype(int),
            })

        return poi
