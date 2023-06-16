import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.main import GazeDetection
import cv2
import csv
import numpy as np
import time


BIOID_DATASET_PATH = "/BioID-FaceDatabase-V1.2/"
N_IMAGES = 1521

gaze_detection_pupil_means = GazeDetection(
    pupil_detection_mode="grad_means", video=False, print_on_serial=False)
gaze_detection_pupil_filtering = GazeDetection(
    pupil_detection_mode="filtering", video=False, print_on_serial=False)


time_means = 0
time_filtering = 0


# number of samples, method, eye, coordindates - methods are resp: ground truth, means of gradients, filtering
pupils_position = np.empty((N_IMAGES, 3, 2, 2))


for index in range(0, N_IMAGES):
    try:
        image = f"{BIOID_DATASET_PATH}BioID_{index:04}.pgm"
        frame = cv2.imread(image)

        start_time = time.time()
        _, gaze_facing, returned_info_means = gaze_detection_pupil_means.detect(
            frame, True)
        time_means += (time.time() - start_time)

        start_time = time.time()
        _, gaze_facing, returned_info_filtering = gaze_detection_pupil_filtering.detect(
            frame, True)
        time_filtering += (time.time() - start_time)

        with open(image[:-3]+"eye", "r") as file:
            gt_values = file.readlines()[1].strip().split("\t")

        arr = np.array([
            [[gt_values[0], gt_values[1]], [gt_values[2], gt_values[3]]],
            [returned_info_means["sx_eye"], returned_info_means["dx_eye"]],
            [returned_info_filtering["sx_eye"], returned_info_filtering["dx_eye"]],
        ])
        

    except:
        print(f"WARNING: error computing index {index}\n")

        arr = np.array([
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]]
        ])

    print(f"OK. Computed images: {index}/{N_IMAGES-1}", end="\r")

    pupils_position[index] = arr


print("Time for computing means of gradient for pupil detection: "+str(time_means))
print("Time for computing filtering for pupil detection: "+str(time_filtering))

times = np.array([0, time_means/N_IMAGES, time_filtering/N_IMAGES])


np.savez("test_results", pupils_position=pupils_position, times = times)
