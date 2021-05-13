'''
Util for config
'''
import sys
import logging

from argparse import ArgumentParser
from mouse_controller import MouseController
from input_feeder import InputFeeder

LOG_FILE = 'logs/gaze.log'
CAM_INPUT_TYPE = 'cam'

def set_logging():
    logging.basicConfig(format='%(asctime)s - %(message)s', filename=LOG_FILE, level=logging.DEBUG)

def createMouseController():
    return MouseController('high', 'slow')

def createInputFeeder(cmd_paras):
    if cmd_paras.input_type==CAM_INPUT_TYPE:
        input_file = None
    else:
        input_file = cmd_paras.input_file

    input_feeder=InputFeeder(input_type=cmd_paras.input_type, input_file=input_file)
    return input_feeder

def get_comand_line_parameters():
    parser = ArgumentParser(description='Mouse Pointer Controller Parameters')
    parser.add_argument('--input_file',
                        default='bin/demo.mp4',
                        type=str,
                        help='IP video stream for video capturing')
    parser.add_argument('--input_type',
                        default='video',
                        choices=['cam', 'video'],
                        type=str,
                        help='Type of Stream')
    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'MYRIAD', 'FPGA'],
                        help='Device CPU, GPU, MYRIAD, FPGA')
    parser.add_argument('--fd',
                        default='/home/stefan/project-controller/model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
                        type=str,
                        help='face detection model')
    parser.add_argument('--lr',
                        default='/home/stefan/project-controller/model/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml',
                        type=str,
                        help='facial landmarks model')
    parser.add_argument('--hp',
                        default='/home/stefan/project-controller/model/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml',
                        type=str,
                        help='head pose estimation model')
    parser.add_argument('--ge',
                        default='/home/stefan/project-controller/model/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml',
                        type=str,
                        help='gaze estimation model')

    parameters = parser.parse_args()
    logging.info('Parameters: -->%s<--',parameters)
    return parameters

def get_comand_lineFP16():
    parser = ArgumentParser(description='Mouse Pointer Controller Parameters')
    parser.add_argument('--input_file',
                        default='bin/demo.mp4',
                        type=str,
                        help='IP video stream for video capturing')
    parser.add_argument('--input_type',
                        default='video',
                        choices=['cam', 'video'],
                        type=str,
                        help='Type of Stream')
    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'MYRIAD', 'FPGA'],
                        help='Device CPU, GPU, MYRIAD, FPGA')
    parser.add_argument('--fd',
                        default='/home/stefan/project-controller/model/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml',
                        type=str,
                        help='face detection model')
    parser.add_argument('--lr',
                        default='/home/stefan/project-controller/model/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml',
                        type=str,
                        help='facial landmarks model')
    parser.add_argument('--hp',
                        default='/home/stefan/project-controller/model/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml',
                        type=str,
                        help='head pose estimation model')
    parser.add_argument('--ge',
                        default='/home/stefan/project-controller/model/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml',
                        type=str,
                        help='gaze estimation model')

    parameters = parser.parse_args()
    logging.info('Parameters: -->%s<--',parameters)
    return parameters