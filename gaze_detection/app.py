from components.face_landmark import FaceLandmarkTracking, FaceLandmarkK
from components.pupil_detection_means_gradients import PupilDetection as PupilDetectionMeansGradient
from components.pupil_detection_filtering import PupilDetection as PupilDetectionFiltering
from components.pnp_solver import PnPSolver

import cv2
import numpy as np
import os


FILTERING = "filtering"
GRAD_MEANS = "grad_means"
NO_PUPIL_DETECTION = "no_pupil"

k = FaceLandmarkK()

class GazeDetection():
    def __init__(self, predictor_path: str = os.path.join('resources', 'predictors', 'shape_predictor_68_face_landmarks.dat'), 
                 pupil_detection_mode: str = GRAD_MEANS, 
                 video: bool = True, 
                 image_path: str = os.path.join('resources', 'images', 'face1.png')) -> None:
        
        self.landmark_tracking = FaceLandmarkTracking(predictor_path)

        if (pupil_detection_mode == GRAD_MEANS):
            self.pupil_detection = PupilDetectionMeansGradient()
        elif(pupil_detection_mode == FILTERING):
            self.pupil_detection = PupilDetectionFiltering()

        # to use the calibration (np.load('calib_results.npz'))
        self.pnp_solver = PnPSolver()

        self.face_facing = False
        self.gaze_facing = False

        if video:
            vid = cv2.VideoCapture(0)
            while(True):
                ret, frame = vid.read()

                self.tracking(frame, pupil_detection_mode)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vid.release()
        else:
            frame = cv2.imread(image_path)
            frame = self.tracking(frame, pupil_detection_mode)
            cv2.imwrite(os.path.splitext(image_path)[
                        0] + '_edited' + os.path.splitext(image_path)[1], frame)
            cv2.waitKey()

    def tracking(self, frame: np.ndarray, pupil_detection_mode: str):

        framebg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for face in self.landmark_tracking.face_analysis(framebg):

            image_points = np.array([
                face.get(k.NOSE),
                face.get(k.CHIN),
                face.get(k.EYE_SX_OUT),
                face.get(k.EYE_DX_OUT),
                face.get(k.MOUTH_SX),
                face.get(k.MOUTH_DX),
            ], dtype="double")

            nose_end_point2D, pitch, yaw, roll = self.pnp_solver.pose(
                frame.shape, image_points)

            face_facing = False
            face_facing_sensibility = 20

            if abs(pitch) < face_facing_sensibility and abs(yaw) < face_facing_sensibility:
                face_facing = True

            eye_frame_horizontal_padding = 0
            eye_frame_vertical_padding = 2

            eyes = [framebg[
                face.get(k.EYE_SX_TOP)[1] - eye_frame_vertical_padding:
                    face.get(k.EYE_SX_BOTTOM)[1] + eye_frame_vertical_padding,
                face.get(k.EYE_SX_IN)[0] - eye_frame_horizontal_padding:
                    face.get(k.EYE_SX_OUT)[0] + eye_frame_horizontal_padding],
                    framebg[
                face.get(k.EYE_DX_TOP)[1]-eye_frame_vertical_padding:
                    face.get(k.EYE_DX_BOTTOM)[1]+eye_frame_vertical_padding,
                face.get(k.EYE_DX_OUT)[0]-eye_frame_horizontal_padding:
                    face.get(k.EYE_DX_IN)[0]+eye_frame_horizontal_padding]
                    ]

            pupil_sx_x, pupil_sx_y = self.pupil_detection.detect_pupil(eyes[0])
            pupil_dx_x, pupil_dx_y = self.pupil_detection.detect_pupil(eyes[1])

            pupil_sx_y, pupil_sx_x = face.get(k.EYE_SX_TOP)[
                1] - eye_frame_vertical_padding + pupil_sx_y, face.get(k.EYE_SX_IN)[0] - eye_frame_horizontal_padding + pupil_sx_x
            pupil_dx_y, pupil_dx_x = face.get(k.EYE_DX_TOP)[
                1] - eye_frame_vertical_padding + pupil_dx_y, face.get(k.EYE_DX_OUT)[0] - eye_frame_horizontal_padding + pupil_dx_x

            # horizzontal ratio that expresses how centered the pupil is within the eyes, from -0.5 to 0.5, 0 is center.
            pupil_sx_center_h_ratio = round((pupil_sx_x - face.get(k.EYE_SX_IN)[0]) / (
                face.get(k.EYE_SX_OUT)[0] - face.get(k.EYE_SX_IN)[0]) - 0.5, 2)
            pupil_dx_center_h_ratio = round((pupil_dx_x - face.get(k.EYE_DX_OUT)[0]) / (
                face.get(k.EYE_DX_IN)[0] - face.get(k.EYE_DX_OUT)[0]) - 0.5, 2)
            gaze_facing = False

            if face_facing:
                gaze_facing = True
            elif yaw < 0 and abs(pupil_sx_center_h_ratio * 100 - yaw) < face_facing_sensibility:
                gaze_facing = True
            elif yaw > 0 and abs(pupil_dx_center_h_ratio * 100 - yaw) < face_facing_sensibility:
                gaze_facing = True

            try:
                cv2.rectangle(frame, (face.get(k.BOX)[0], face.get(k.BOX)[1]), (face.get(k.BOX)[
                    0]+face.get(k.BOX)[2], face.get(k.BOX)[1]+face.get(k.BOX)[3]), (255, 0, 255), 2)

                cv2.rectangle(frame, (face.get(k.EYE_SX_IN)[0]-eye_frame_horizontal_padding,
                                      face.get(k.EYE_SX_TOP)[1]-eye_frame_vertical_padding),
                              (face.get(k.EYE_SX_OUT)[0]+eye_frame_horizontal_padding,
                               face.get(k.EYE_SX_BOTTOM)[1]+eye_frame_vertical_padding),
                              (255, 0, 255), 2)
                cv2.rectangle(frame, (face.get(k.EYE_DX_OUT)[0]-eye_frame_horizontal_padding,
                                      face.get(k.EYE_DX_TOP)[1]-eye_frame_vertical_padding),
                              (face.get(k.EYE_DX_IN)[0]+eye_frame_horizontal_padding,
                               face.get(k.EYE_DX_BOTTOM)[1]+eye_frame_vertical_padding),
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

                cv2.putText(frame, f"Pupil horizzontal ratios: {pupil_dx_center_h_ratio}, {pupil_sx_center_h_ratio}", (face.get(k.EYE_DX_OUT)[0], 160),
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
                    cv2.putText(frame, "Roll: " + str(round(roll)), (face.get(k.EYE_DX_OUT)[0], 100),
                                1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "Pitch: " + str(round(pitch)), (face.get(k.EYE_DX_OUT)[0], 120),
                                1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "Yaw: " + str(round(yaw)), (face.get(k.EYE_DX_OUT)[0], 140),
                                1, 1, (255, 255, 255), 1, cv2.LINE_AA)

            except:
                print("Error during info display")

            try:
                cv2.putText(frame, "Face facing camera: " + str(int(face_facing)), (face.get(k.EYE_DX_OUT)[0], 180),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(frame, "Gaze facing camera: " + str(int(gaze_facing)), (face.get(k.EYE_DX_OUT)[0], 200),
                            1, 1, (255, 255, 255), 1, cv2.LINE_AA)
            except:
                print("Error during info display")

        cv2.imshow('frame',  frame)
        return frame


g = GazeDetection()
