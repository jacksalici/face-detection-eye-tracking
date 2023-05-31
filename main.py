from components.face_detection import HaarCascade
from components.face_landmark import FaceLandmarkTracking
from components.pupil_detection import PupilDetection
from components.pnp_solver import PnPSolver

import cv2
import numpy as np


vid = cv2.VideoCapture(0)

landmark_tracking = FaceLandmarkTracking()
pupil_detection = PupilDetection()
# np.load('calib_results.npz')
pnp_solver = PnPSolver()

while(True):
    ret, frame = vid.read()
    #frame = cv2.imread("rdg.jpg")
    framebg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for face in landmark_tracking.face_analysis(framebg):
        image_points = np.array([
            face.get("nose"),
            face.get("chin"),
            face.get("eye_sx_out"),
            face.get("eye_dx_out"),
            face.get("mouth_sx"),
            face.get("mouth_dx"),
        ], dtype="double")

        nose_end_point2D, pitch, yaw, roll = pnp_solver.pose(
            frame.shape, image_points)

        eye_frame_padding = 5
        cv2.rectangle(frame, (face.get("eye_sx_in")[0]-eye_frame_padding, 
                              face.get("eye_sx_top")[1]-eye_frame_padding),
                      (face.get("eye_sx_out")[0]+eye_frame_padding, 
                       face.get("eye_sx_bottom")[1]+eye_frame_padding),
                      (255, 0, 255), 1)
        cv2.rectangle(frame, (face.get("eye_dx_out")[0]-eye_frame_padding, 
                              face.get("eye_dx_top")[1]-eye_frame_padding),
                      (face.get("eye_dx_in")[0]+eye_frame_padding, 
                       face.get("eye_dx_bottom")[1]+eye_frame_padding),
                      (255, 0, 255), 1)

        (pupil_sx_x, pupil_sx_y) = pupil_detection.detect_pupil(framebg[
            face.get("eye_sx_top")[1] - eye_frame_padding: 
                face.get("eye_sx_bottom")[1] + eye_frame_padding,
            face.get("eye_sx_in")[0] - eye_frame_padding: 
                face.get("eye_sx_out")[0] + eye_frame_padding])

        pupil_sx_y, pupil_sx_x = face.get("eye_sx_top")[1] - eye_frame_padding +pupil_sx_y, face.get("eye_sx_in")[0]  - eye_frame_padding + pupil_sx_x

        (pupil_dx_x, pupil_dx_y) = pupil_detection.detect_pupil(framebg[
            face.get("eye_dx_top")[1]-eye_frame_padding: 
                face.get("eye_dx_bottom")[1]+eye_frame_padding,
            face.get("eye_dx_out")[0]-eye_frame_padding: 
                face.get("eye_sx_in")[0]+eye_frame_padding])

        pupil_dx_y, pupil_dx_x = face.get("eye_dx_top")[1] - eye_frame_padding +pupil_dx_y, face.get("eye_dx_out")[0]  - eye_frame_padding + pupil_dx_x

        cv2.circle(frame, (pupil_sx_x, pupil_sx_y),
                   10, (0, 255, 255), 1)

        cv2.circle(frame, (pupil_dx_x, pupil_dx_y),
                   10, (0, 255, 255), 1)

        for p in list(face.values())[1:]:
            cv2.circle(frame, (int(p[0]), int(p[1])),
                       2, (255, 255, 0), -1)

        try:
            frame = cv2.line(frame, tuple(image_points[0].ravel().astype(int)), tuple(
                nose_end_point2D[0].ravel().astype(int)), (255, 0, 0), 3)
            frame = cv2.line(frame, tuple(image_points[0].ravel().astype(int)), tuple(
                nose_end_point2D[1].ravel().astype(int)), (0, 255, 0), 3)
            frame = cv2.line(frame, tuple(image_points[0].ravel().astype(int)), tuple(
                nose_end_point2D[2].ravel().astype(int)), (0, 0, 255), 3)

            if roll and pitch and yaw:
                cv2.putText(frame, "Roll: " + str(round(roll)), (face.get("eye_dx_out")[0], 100),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Pitch: " + str(round(pitch)), (face.get("eye_dx_out")[0], 120),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Yaw: " + str(round(yaw)), (face.get("eye_dx_out")[0], 140),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            print(Exception.with_traceback)

    cv2.imshow('frame',  frame)
    # cv2.waitKey(10000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
