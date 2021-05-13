import os
import time
import logging
import cv2
import numpy as np

from openvino.inference_engine import IECore
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
DEFAULT_DEVICE = 'CPU'


class BaseModel:
    '''
    BaseModelClass
    '''

    def __init__(self, model_name, device=DEFAULT_DEVICE, plugin=None):
        self.device = device
        self.model_name = model_name

        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        logging.debug('BaseModel __init__ device    -->%s<--', device)
        logging.debug('BaseModel __init__ model_xml -->%s<--', model_xml)
        logging.debug('BaseModel __init__ model_bin -->%s<--', model_bin)

        # NC2 Stick can only hold one instance
        if not plugin:
            plugin = IECore()

        try:
            net = plugin.read_network(model=model_xml, weights=model_bin)
            self.input_name = next(iter(net.input_info))
            self.input_shape = net.input_info[self.input_name].input_data.shape
            self.output_name = next(iter(net.outputs))
            self.output_shape = net.outputs[self.output_name].shape
        except Exception as e:
            logging.critical(e, exc_info=True)
            raise ValueError(
                'Problem init Inference Engine -->%s<--'.format(str(e)))

        self.core = plugin
        self.net = net
        logging.debug(
            'BaseModel Inference Engine successfully init {}'.format(model_xml))

    def load_model(self):
        try:
            start_time = cv2.getTickCount()
            self.net = self.core.load_network(
                network=self.net, device_name=self.device, num_requests=0)
            end_time = cv2.getTickCount()
            logging.debug('Load model {} : {:.4f} seconds'.format(os.path.basename(self.model_name),
                                                                  (end_time-start_time)/cv2.getTickFrequency()))

        except Exception as e:
            logging.critical(e, exc_info=True)
            raise ValueError('Can not load the model. {}'.format(e))

    def exec_infer(self, image):
        output_img = self.preprocess_input(image)
        try:
            start_time = cv2.getTickCount()
            result = self.net.infer({self.input_name: output_img})
            end_time = cv2.getTickCount()
            logging.debug('RESULT EXEC_INFER {} : {:.4f} seconds'.format(os.path.basename(self.model_name),
                                                                         (end_time-start_time)/cv2.getTickFrequency()))
        except Exception as e:
            logging.critical(e, exc_info=True)
            raise ValueError("Can not execute infer request.\n {}".format(e))
        return result

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, input_image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
