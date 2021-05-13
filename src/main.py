import os
import logging
import cv2
import time
import numpy as np
import pyautogui

# Helper/Util Classes
from input_feeder import InputFeeder
from mouse_controller import MouseController
from config_util import get_comand_line_parameters, set_logging, createMouseController, createInputFeeder, get_comand_lineFP16
from image_helper import crop_eyes, plot_gaze

# Modell Classes
from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarksDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimatior
from mouse_controller import MouseController


from openvino.inference_engine import IECore

# Dectors and Estimators as global variables
face_detector = None
landmark_dectector = None
head_pose_estimator = None
gaze_estimator = None
mouse_controller = None


def pipline(input_image, i):
    global face_detector, landmark_dectector, head_pose_estimator, gaze_estimator, mouse_controller
    pipline_start_time = cv2.getTickCount()
    logging.debug('starts run_pipline {:d}'.format(i))

    logging.debug('run face_dector.predict_face')
    face, upper_corner, pf_image = face_detector.predict_face(input_image)
    logging.debug('face detected finished')
    if np.all(face.shape):

        logging.debug('run landmark_dector.predict_eyes')
        face, left_eye, right_eye = landmark_dectector.predict_eyes(face)

        logging.debug('run crop_eyes')
        crop_left, crop_right = crop_eyes(
            pf_image, upper_corner, (left_eye, right_eye))

        logging.debug('run head_pose_estimator.predict_vector')
        head_pos_angels = head_pose_estimator.predict_vector(face)

        logging.debug('run gaze_estimator.predict_gaze')
        gaze = gaze_estimator.predict_gaze(
            crop_left, crop_right, head_pos_angels)

        logging.debug('run plot_gaze')
        face = plot_gaze(face, gaze, left_eye, right_eye)
        gaze_position = gaze[0], gaze[1]
    else:
        logging.info("No person detected, Application is closing ...")
    
    pipline_end_time = cv2.getTickCount()
    logging.info('Pipline Runtime: {:.4f} seconds. Run {:d}'.format(
        (pipline_end_time-pipline_start_time)/cv2.getTickFrequency(), i))
    return face, gaze_position


def main():
    global face_detector, landmark_dectector, head_pose_estimator, gaze_estimator, mouse_controller
    # Load parameters
    app_start_time = cv2.getTickCount()

    set_logging()
    logging.info('start APP')
    pyautogui.FAILSAFE = False
    cmd_paras = get_comand_line_parameters()
    #cmd_paras = get_comand_lineFP16()

    # Setup Projects
    mouse_controller = createMouseController()

    # Setup Classes
    logging.info('setUp dectors and estimators started')

    # NC2 can only handel one instance of IECore
    plugin = IECore()
    face_detector = FaceDetector(cmd_paras.fd, cmd_paras.device, plugin)
    face_detector.load_model()
    landmark_dectector = FacialLandmarksDetector(
        cmd_paras.lr, cmd_paras.device, plugin)
    landmark_dectector.load_model()
    head_pose_estimator = HeadPoseEstimator(
        cmd_paras.hp, cmd_paras.device, plugin)
    head_pose_estimator.load_model()
    gaze_estimator = GazeEstimatior(cmd_paras.ge, cmd_paras.device, plugin)
    gaze_estimator.load_model()
    logging.info('setUp dectors estimators ends')

    logging.info('start inputFeeder read stream')
    input_feeder = createInputFeeder(cmd_paras)
    input_feeder.load_data()

    # RUN the pipline
    i = 0
    try:
        for tmp_image in input_feeder.next_batch():
            if tmp_image is not None:
                i = i+1
                tmp_image, gaze_position = pipline(tmp_image, i)
                cv2.imshow('frame', tmp_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                mouse_controller.move(gaze_position[0], gaze_position[1])
            else:
                break
    finally:
        input_feeder.close()
        logging.info('end inputFeeder end stream')
    
    app_end_time = cv2.getTickCount()
    logging.info('App Runtime: {:.4f} seconds pipline runs {:d}'.format(
        (app_end_time-app_start_time)/cv2.getTickFrequency(), i))


if __name__ == "__main__":
    main()
