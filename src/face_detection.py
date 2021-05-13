'''
Face detection model.
'''
import cv2
import logging
import numpy as np

from model import BaseModel
from image_helper import draw_face_box, crop_face

DEFAULT_THREDHOLD = 0.6
DEFAULT_HEIGHT = 384
DEFAULT_WIGHT = 672


class FaceDetector(BaseModel):
    '''
    Class for the Face Detection Model.
    Inherit from BaseModel.
    '''

    def preprocess_output(self, outputs):
        content = outputs[0][0]
        face = content[0]
        if face[1] == 0:
            return []
        for tmp in content[1:]:
            if tmp[1] == 0:
                return face
            if tmp[2] > face[2]:
                face = tmp
        return face

    def predict_face(self, image):
        output_face = self.predict(image)
        output_image = draw_face_box(image, output_face, DEFAULT_THREDHOLD)
        face_crop, upper_corner = crop_face(output_image, output_face)
        return face_crop, upper_corner, output_image

    def predict(self, image):
        outputs = self.exec_infer(image)
        return self.preprocess_output(outputs[self.output_name])

    def preprocess_input(self, input_image):
        logging.debug('preprocess_input FaceDector start')
        image = np.copy(input_image)
        try:
            image = cv2.resize(
                image, (DEFAULT_WIGHT, DEFAULT_HEIGHT), cv2.INTER_CUBIC)
        except TypeError:
            return None

        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, DEFAULT_HEIGHT, DEFAULT_WIGHT)

        logging.debug('preprocess_input ends')
        return image
