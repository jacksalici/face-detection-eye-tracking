import cv2, os




class HaarCascade():
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_frontalface_alt.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_eye.xml'))

    def face_detection(self, frame, scale_factor=1.2, min_neighbors=2):            
        return self.face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale_factor, min_neighbors)
    
    
    def face_framing(self, faces, frame):
        for (x, y, w, h) in faces:            
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return frame
        
        
    
    def eye_framing(self, faces, frame):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyeglasses = self.eye_cascade.detectMultiScale(roi_gray)

            
            
            for (ex, ey, ew, eh) in eyeglasses:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                #cv2.circle(roi_color, ((ex+ew)/2, (ey+eh)/2), (ew+eh)/4, (0, 0, 255), 4 )

        return frame










# define a video capture object
vid = cv2.VideoCapture(0)
cascade = HaarCascade()

while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    cv2.imshow('frame', cascade.eye_framing(cascade.face_detection(frame), frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
