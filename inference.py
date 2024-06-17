import keras
import cv2


class Inference:
    def __init__(self,image):
        self.image = image

    def predict(self):
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        model = keras.models.load_model()

