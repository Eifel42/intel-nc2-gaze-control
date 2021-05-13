import cv2
import logging
import numpy as np

from model import BaseModel

DEFAULT_SIDE = 60


class HeadPoseEstimator(BaseModel):

    def preprocess_output(self, outputs):
        angle_yaw = outputs['angle_y_fc']
        angle_pitch = outputs['angle_p_fc']
        angle_roll = outputs['angle_r_fc']
        return np.concatenate([angle_yaw, angle_pitch, angle_roll], axis=1)

    def predict_vector(self, image):
        outputs = self.exec_infer(image)
        head_pose_angles = self.preprocess_output(outputs)
        return head_pose_angles

    def preprocess_input(self, input_image):
        logging.debug('preprocess_input HeadPosEstimator start')
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
