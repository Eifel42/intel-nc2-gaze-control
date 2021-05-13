'''
Facial landmarks detection model.
'''
import os
import cv2
import time
import logging
import numpy as np

from model import BaseModel
from image_helper import draw_eyes_boxes, get_eyes_coordinates

DEFAULT_SIDE = 48


class FacialLandmarksDetector(BaseModel):

    def exec_infer(self, input_image):
        tmp_image = self.preprocess_input(input_image)
        try:
            start_time = cv2.getTickCount()
            result = self.net.infer({self.input_name: tmp_image})[
                self.output_name]
            end_time = cv2.getTickCount()
            logging.debug('RESULT EXEC_INFER {} : {:.4f} seconds'.format(os.path.basename(self.model_name),
                                                                         (end_time-start_time)/cv2.getTickFrequency()))
        except Exception as e:
            logging.critical(e, exc_info=True)
            raise ValueError("Can not execute infer request.\n {}".format(e))
        return result
        
    def preprocess_output(self, outputs):
        arr = outputs.flatten()
        eyes = arr[:4]
        return eyes

    def predict_eyes(self, input_image):
        outputs = self.exec_infer(input_image)
        eyes = self.preprocess_output(outputs)
        image = draw_eyes_boxes(input_image, eyes)
        left_eye, right_eye = get_eyes_coordinates(image, eyes)
        return image, left_eye, right_eye

    def preprocess_input(self, input_image):
        logging.debug('preprocess_input landmark start')
        image = input_image.copy()
        try:
            image = cv2.resize(
                image, (DEFAULT_SIDE, DEFAULT_SIDE), cv2.INTER_AREA)
        except TypeError:
            return None

        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, DEFAULT_SIDE, DEFAULT_SIDE)

        logging.debug('preprocess_input ends')
        return image
