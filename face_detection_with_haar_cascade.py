import cv2, os




class HaarCascade():
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_frontalface_alt.xml'))
        self.eye_cascade = cv2.CascadeClassifier(os.path.join('predictors','haarcascade_eye.xml'))

    def face_detection(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
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
    
    cv2.imshow('frame', cascade.face_detection(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
