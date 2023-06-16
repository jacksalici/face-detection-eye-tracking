from src.main import GazeDetection

g=GazeDetection(print_on_serial=True, annotate_image=True, pupil_detection_mode="grad_means", serial_writing_step=7, crop_frame=False)