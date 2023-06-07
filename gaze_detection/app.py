from components.face_landmark import FaceLandmarkTracking, FaceLandmarkK
from components.pupil_detection_means_gradients import PupilDetection as PupilDetectionMeansGradient
from components.pupil_detection_filtering import PupilDetection as PupilDetectionFiltering
from components.pnp_solver import PnPSolver

import cv2
import numpy as np
import os
import serial
import struct

FILTERING = "filtering"
GRAD_MEANS = "grad_means"
NO_PUPIL_DETECTION = "no_pupil"

k = FaceLandmarkK()

class GazeDetection():
    def __init__(self, predictor_path: str = os.path.join('resources', 'predictors', 'shape_predictor_68_face_landmarks.dat'), 
                 pupil_detection_mode: str = GRAD_MEANS, 
                 video: bool = True, 
                 image_path: str = None,
                 visual_verbose: bool = False,
                 save_image: bool = False,
                 print_on_serial: bool = True,
                 serial_port: str = "/dev/tty.usbmodem1201",
                 annotate_image: bool = False,
                 crop_frame: bool = True,
                 crop_frame_paddings: tuple = (0.5,0,0.15,0) #top, right, bottom, left / [0:1]
                 ) -> None:
        
        self.landmark_tracking = FaceLandmarkTracking(predictor_path)

        self.pupil_detection_mode = pupil_detection_mode
        if (pupil_detection_mode == GRAD_MEANS):
            self.pupil_detection = PupilDetectionMeansGradient()
        elif(pupil_detection_mode == FILTERING):
            self.pupil_detection = PupilDetectionFiltering(visual_verbose)
        else:
            self.pupil_detection = None

        # to use the calibration (np.load('calib_results.npz'))
        self.pnp_solver = PnPSolver()


        self.face_facing = False
        self.gaze_facing = False
        
        self.print_on_serial = print_on_serial
        self.annotate_image = annotate_image
        
        self.crop_frame = crop_frame
        self.crop_frame_paddings = crop_frame_paddings
        
        try:
            if print_on_serial:
                self.serial_port = serial.Serial(serial_port, 9600)
        except:
            print_on_serial = False
            

        if video:
            vid = cv2.VideoCapture(0)
            while(True):
                ret, frame = vid.read()

                return_frame, _ = self.tracking(frame)

                cv2.imshow('frame',  return_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            vid.release()
        else:
            if image_path is None:
                frame = cv2.imread(image_path)
                
                frame = self.tracking(frame)
                
                if save_image:
                    cv2.imwrite(os.path.splitext(image_path)[
                                0] + '_edited' + os.path.splitext(image_path)[1], frame)
                cv2.waitKey()

    def tracking(self, frame: np.ndarray):

        framebg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.landmark_tracking.face_analysis(framebg)
        
        biggest_face_index = 0
        biggest_face_area = 0

        for face_index, face in enumerate (faces):

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
            gaze_facing = face_facing
            
            if self.pupil_detection != None:
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
            
                if face_facing:
                    gaze_facing = True
                elif yaw < 0 and abs(pupil_sx_center_h_ratio * 100 - yaw) < face_facing_sensibility:
                    gaze_facing = True
                elif yaw > 0 and abs(pupil_dx_center_h_ratio * 100 - yaw) < face_facing_sensibility:
                    gaze_facing = True

            
            if self.annotate_image:
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
                    
                    for p in list(face.values())[1:]:
                        cv2.circle(frame, (int(p[0]), int(p[1])),
                                2, (255, 255, 0), -1)
                except:
                    print("WARNING: Error during info display")

                if (self.pupil_detection != None):
                    try:
                        cv2.circle(frame, (pupil_sx_x, pupil_sx_y),
                                10, (0, 255, 255), 2)

                        cv2.circle(frame, (pupil_dx_x, pupil_dx_y),
                                10, (0, 255, 255), 2)

                        cv2.putText(frame, f"Pupil horizzontal ratios: {pupil_dx_center_h_ratio}, {pupil_sx_center_h_ratio}", (face.get(k.EYE_DX_OUT)[0], 160),
                                    1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    except:
                        print("WARNING: Error during info display")

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
                    print("WARNING: Error during info display")

                try:
                    cv2.putText(frame, "Gaze facing camera: " + str(int(gaze_facing)), (face.get(k.EYE_DX_OUT)[0], 200),
                                1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                except:
                    print("WARNING: Error during info display")
                
                try:
                    cv2.putText(frame, "Face facing camera: " + str(int(face_facing)), (face.get(k.EYE_DX_OUT)[0], 180),
                                1, 1, (255, 255, 255), 1, cv2.LINE_AA)
                except:
                    print("WARNING: Error during info display")

            
            #check the area of each face and find the max one
            current_face_area = face.get(k.BOX)[2]*face.get(k.BOX)[3]
            if current_face_area > biggest_face_area:
                biggest_face_area = current_face_area
                biggest_face_index = face_index
                
        if len(faces)>0:
            
            frame_center = frame.shape[1]/2
            (fx, fy, fw, fh) = faces[biggest_face_index].get(k.BOX) 
            face_center = fx + fw/2 
            
            try:
                if self.print_on_serial:
                    if face_center<frame_center-100:
                        self.serial_port.write(struct.pack('f', 5))
                    elif face_center>frame_center+100:
                        self.serial_port.write(struct.pack('f', -5))
            except:
                print("WARNING: Error writing on serial")
                
            
            if self.crop_frame:
                
                return frame[max(int(fy-fh*self.crop_frame_paddings[0]),0):min(fy+int(fh*(1+self.crop_frame_paddings[2])), frame.shape[0]), 
                             max(int(fx-fw*self.crop_frame_paddings[3]),0):min(fx+int(fw*(1+self.crop_frame_paddings[1])), frame.shape[1])], gaze_facing
            else:
                return frame, gaze_facing
        
        else:
            
            return frame, False


g=GazeDetection(pupil_detection_mode=NO_PUPIL_DETECTION, annotate_image=True, print_on_serial=False)