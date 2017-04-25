import numpy
import cv2
import random

class FaceDetector:
    
    def __init__(self, 
                 cascade='haarcascade_frontalface_alt2.xml',
                 scale_factor=1.1,
                 min_size=(128, 128),
                 min_neighboors=15):
        
        self._cascade = cv2.CascadeClassifier(cascade)
        self._options = {
            'scaleFactor': scale_factor, 
            'minSize': min_size,
            'minNeighbors': min_neighboors
        }
        
    def get_face_image(self, image):
        
        image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)       
 
        faces = self._cascade.detectMultiScale(image, **self._options)
        
        if len(faces) == 0:
             return image

        rect = random.choice(faces)
        x, y, w, h = rect
        
        margin = w // 3
        
        x = x - margin if x > margin else 0
        y = y - (2 * margin)  if y > 2 * margin else 0
        
        w += 2 * margin
        h += 2 * margin
        
        face = image[0:y+h, x:x+w]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
