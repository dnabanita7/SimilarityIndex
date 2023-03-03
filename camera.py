import os
import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        
        model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        while True:
            # read current frame
            _, img = camera.read()
            face  = model.detectMultiScale(img)
            crop_img = img
            if len(face) != 0:
                x1 = face[0][0]
                y1 = face[0][1]
                x2 = face[0][2] + x1
                y2 = face[0][3] + y1 
                crop_img = img[y1:y2 , x1:x2]
                

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', crop_img)[1].tobytes()
