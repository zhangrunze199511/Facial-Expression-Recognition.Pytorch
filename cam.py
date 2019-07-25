import cv2
import numpy as np


class Camera:
    def __init__(self, height, width):
        self.operating = False
        self.height = height
        self.width = width
        self.cap = cv2.VideoCapture(0) # use local camera as video source
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

    def start(self, processingFunction):# processing fuction: the function used to process image, take an image grid as input para
        self.operating = True
        while(self.operating):
            ret, frame = self.cap.read()
            result, img = processingFunction(frame)
            # cv2.imshow('capture', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        return True

    def start_once(self, processingFunction):
        ret, frame = self.cap.read()
        result = processingFunction(frame)
        return result




    def stop(self):
        self.operating = False
        cv2.destroyAllWindows()

    def __delete__(self, instance):
        self.cap.release()
        cv2.destroyAllWindows()



