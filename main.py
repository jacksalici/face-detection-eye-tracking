from components.face_landmark import FaceLandmarkTracking
from components.pupil_detection_2 import PupilDetection
from components.pnp_solver import PnPSolver

import cv2
import numpy as np


vid = cv2.VideoCapture(0)

landmark_tracking = FaceLandmarkTracking()
pupil_detection = PupilDetection()
pnp_solver = PnPSolver()  # to use the calibration (np.load('calib_results.npz'))

face_facing = False
gaze_facing = False

while(True):
    ret, frame = vid.read()  # frame = cv2.imread("face1.png")
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

        face_facing = False
        face_facing_sensibility = 20

        if abs(pitch) < face_facing_sensibility and abs(yaw) < face_facing_sensibility:
            face_facing = True

        eye_frame_padding = 5+2

        eyes = [framebg[
            face.get("eye_sx_top")[1] - eye_frame_padding:
                face.get("eye_sx_bottom")[1] + eye_frame_padding,
            face.get("eye_sx_in")[0] - eye_frame_padding:
                face.get("eye_sx_out")[0] + eye_frame_padding],
                framebg[
            face.get("eye_dx_top")[1]-eye_frame_padding:
                face.get("eye_dx_bottom")[1]+eye_frame_padding,
            face.get("eye_dx_out")[0]-eye_frame_padding:
                face.get("eye_dx_in")[0]+eye_frame_padding]
                ]

        pupil_sx_x, pupil_sx_y = pupil_detection.detect_pupil(eyes[0], 0)
        pupil_dx_x, pupil_dx_y = pupil_detection.detect_pupil(eyes[1], 1)

        if pupil_sx_x and pupil_sx_y:
            pupil_sx_y, pupil_sx_x = face.get("eye_sx_top")[
                1] - eye_frame_padding + pupil_sx_y, face.get("eye_sx_in")[0] - eye_frame_padding + pupil_sx_x
            pupil_dx_y, pupil_dx_x = face.get("eye_dx_top")[
                1] - eye_frame_padding + pupil_dx_y, face.get("eye_dx_out")[0] - eye_frame_padding + pupil_dx_x
            
            
            # horizzontal ratio that expresses how centered the pupil is within the eyes, from -0.5 to 0.5, 0 is center. 
            pupil_sx_center_h_ratio = round((pupil_sx_x - face.get("eye_sx_in")[0]) / (face.get("eye_sx_out")[0] - face.get("eye_sx_in")[0]) - 0.5 ,2) 
            pupil_dx_center_h_ratio = round((pupil_dx_x - face.get("eye_dx_out")[0]) / (face.get("eye_dx_in")[0] - face.get("eye_dx_out")[0]) - 0.5 ,2)

            gaze_facing = False
            gaze_facing_direction = (pupil_dx_center_h_ratio + pupil_sx_center_h_ratio) #double of the mean between the directions [-1, 1]
            gaze_facing_direction *= 50 #angle of view
            
            if face_facing or abs(gaze_facing_direction - yaw)<face_facing_sensibility:
                gaze_facing = True
            
        
        
        try:
            cv2.rectangle(frame, (face.get("box")[0], face.get("box")[1]), (face.get("box")[
                          0]+face.get("box")[2], face.get("box")[1]+face.get("box")[3]), (255, 0, 255), 2)

            cv2.rectangle(frame, (face.get("eye_sx_in")[0]-eye_frame_padding,
                                  face.get("eye_sx_top")[1]-eye_frame_padding),
                          (face.get("eye_sx_out")[0]+eye_frame_padding,
                           face.get("eye_sx_bottom")[1]+eye_frame_padding),
                          (255, 0, 255), 2)
            cv2.rectangle(frame, (face.get("eye_dx_out")[0]-eye_frame_padding,
                                  face.get("eye_dx_top")[1]-eye_frame_padding),
                          (face.get("eye_dx_in")[0]+eye_frame_padding,
                           face.get("eye_dx_bottom")[1]+eye_frame_padding),
                          (255, 0, 255), 2)
        except:
            print("Error during info display")
        
        try:
            cv2.circle(frame, (pupil_sx_x, pupil_sx_y),
                       10, (0, 255, 255), 2)

            cv2.circle(frame, (pupil_dx_x, pupil_dx_y),
                       10, (0, 255, 255), 2)

            for p in list(face.values())[1:]:
                cv2.circle(frame, (int(p[0]), int(p[1])),
                           2, (255, 255, 0), -1)
            
            cv2.putText(frame, f"Pupil horizzontal ratios: {pupil_dx_center_h_ratio}, {pupil_sx_center_h_ratio} - {gaze_facing_direction}°", (face.get("eye_dx_out")[0], 160),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
        except:
            print("Error during info display")
        
        try:
            frame = cv2.line(frame, tuple(image_points[0].ravel().astype(int)), tuple(
                nose_end_point2D[0].ravel().astype(int)), (255, 0, 0), 2)
            frame = cv2.line(frame, tuple(image_points[0].ravel().astype(int)), tuple(
                nose_end_point2D[1].ravel().astype(int)), (0, 255, 0), 2)
            frame = cv2.line(frame, tuple(image_points[0].ravel().astype(int)), tuple(
                nose_end_point2D[2].ravel().astype(int)), (0, 0, 255), 2)

            if roll and pitch and yaw:
                cv2.putText(frame, "Roll: " + str(round(roll)) + "°", (face.get("eye_dx_out")[0], 100),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Pitch: " + str(round(pitch))  + "°", (face.get("eye_dx_out")[0], 120),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Yaw: " + str(round(yaw))  + "°", (face.get("eye_dx_out")[0], 140),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)

        except:
            print("Error during info display")
            
        try:
            cv2.putText(frame, "Face facing camera: " + str(int(face_facing)), (face.get("eye_dx_out")[0], 190),
                        0, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(frame, "Gaze facing camera: " + str(int(gaze_facing)), (face.get("eye_dx_out")[0], 220),
                        0, 1, (255, 255, 255), 1, cv2.LINE_AA)
        except:
            print("Error during info display")

    cv2.imshow('frame',  frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()