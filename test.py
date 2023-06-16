from src.main import GazeDetection
import cv2
import csv
import numpy as np
import time


BIOID_DATASET_PATH = "/Users/jacksalici/Downloads/BioID-FaceDatabase-V1.2/"
N_IMAGES = 1521

gaze_detection_pupil_means = GazeDetection(pupil_detection_mode="grad_means", video=False, print_on_serial=False)
gaze_detection_pupil_filtering = GazeDetection(pupil_detection_mode="filtering", video=False, print_on_serial=False)




time_means = 0
time_filtering = 0

for index in range(0,N_IMAGES):
    try:
        image = f"{BIOID_DATASET_PATH}BioID_{index:04}.pgm"
        frame = cv2.imread(image)
        
        start_time = time.time()
        _, gaze_facing, returned_info1 = gaze_detection_pupil_filtering.detect(frame, True)
        time_filtering+= (time.time() - start_time)
        
        start_time = time.time()
        _, gaze_facing, returned_info2 = gaze_detection_pupil_means.detect(frame, True)
        time_means+= (time.time() - start_time)
        
        with open(image[:-3]+"eye", "r") as file:
            gt_values = file.readlines()[1].strip().split("\t")
            
    except:
        print(f"WARNING: error computing index {index}")
    print(f"OK. Computed images: {index}/{N_IMAGES}")
    

print ("Time for computing means of gradient for pupil detection: "+str(time_means))
print ("Time for computing filtering for pupil detection: "+str(time_means))