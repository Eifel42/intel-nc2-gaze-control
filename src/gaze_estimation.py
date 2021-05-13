'''
gaze estimation model.
'''
import os
import time
import cv2
import logging
import numpy as np

from model import BaseModel

DEFAULT_SIDE = 60


class GazeEstimatior(BaseModel):

    def exec_infer(self, left_eye, right_eye, head_pose_angles):
        try:
            pre_left_eye = self.preprocess_input(left_eye)
            pre_right_eye = self.preprocess_input(right_eye)
            infer_input = {
                'left_eye_image': pre_left_eye,
                'right_eye_image': pre_right_eye,
                'head_pose_angles': head_pose_angles
            }

            start_time = cv2.getTickCount()
            result = self.net.infer(infer_input)
            end_time = cv2.getTickCount()
            logging.debug('RESULT EXEC_INFER {} : {:.4f} seconds'.format(os.path.basename(self.model_name),
                                                                         (end_time-start_time)/cv2.getTickFrequency()))
        except Exception as e:
            logging.critical(e, exc_info=True)
            raise ValueError("Can not execute infer request.\n {}".format(e))
        return result

    def preprocess_output(self, outputs):
        gaze_output = outputs['gaze_vector']
        return gaze_output

    def predict_gaze(self, left_eye, right_eye, head_pose_angles):
        logging.debug('GazeEstimator.predict gaze started')
        try:
            outputs = self.exec_infer(left_eye, right_eye, head_pose_angles)
            gaze_output = self.preprocess_output(outputs)
            logging.debug(gaze_output)

            logging.debug('preprocess_input ends')
            return gaze_output.flatten()
        except Exception as e:
            logging.critical(e, exc_info=True)
            return []

    def preprocess_input(self, input_image):
        output = np.copy(input_image)
        output = output.transpose((2, 0, 1))
        output = output.reshape(1, 3, DEFAULT_SIDE, DEFAULT_SIDE)
        return output
