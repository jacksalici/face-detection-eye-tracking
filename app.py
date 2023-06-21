from src.main import GazeDetection

g=GazeDetection(image_path='src/resources/images/face1.png', visual_verbose=True, print_on_serial=False, annotate_image=True, pupil_detection_mode="filtering", serial_writing_step=7, crop_frame=False, save_image=True, video=True)