from components.face_detection import HaarCascade
from components.face_landmark import FaceLandmarkTracking
from components.pupil_detection import PupilDetection
from components.pnp_solver import PnPSolver

import cv2
import numpy as np


vid = cv2.VideoCapture(0)
landmark_tracking = FaceLandmarkTracking()
pupil_detection = PupilDetection()
pnp_solver = PnPSolver()

while(True):
    ret, frame = vid.read()
    #frame = cv2.imread("rdg.jpg")
    framebg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for face in landmark_tracking.face_analysis(framebg):
        image_points = np.array([
        face.get("traits").get("nose"),
        face.get("traits").get("chin"),
        face.get("eye_corners").get("sx_out"),
        face.get("eye_corners").get("dx_out"),
        face.get("traits").get("mouth_sx"),
        face.get("traits").get("mouth_dx"),
    ], dtype="double")
    
    
        nose_end_point2D = pnp_solver.pose(frame.shape, image_points)
            
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255,0,0), 2)


    cv2.imshow('frame',  frame)
    #cv2.waitKey(10000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
