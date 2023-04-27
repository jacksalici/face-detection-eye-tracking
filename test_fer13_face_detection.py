from face_detection_with_haar_cascade import HaarCascade
import cv2
from pathlib import Path

"""
    Script to test how good the haarCascade OpenCv detector works using the Fer2013 dataset.
    With my current settings, the results are following. 
    {'0': 18016, '1': 17867, '2': 4}
    Please note that the value of each key represents the number of images in which that key number of faces have been detected.
"""

FER2013_PATH = "fer2013"
haarCascade = HaarCascade()
face_detected = {'0':0, '1':0, '2':0}

for image in list(Path(FER2013_PATH).rglob("*.jpg")):
    faces = haarCascade.face_detection(cv2.imread(str(image)))
    face_detected[str(len(faces))]+=1

print(f"\nTOTALS:\n{face_detected}")
