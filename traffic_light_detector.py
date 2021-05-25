import sys
import os
import json

# Script level imports
from traffic_light_detection_module.yolo import YOLO


class TrafficLightDetector:

    def __init__(self):
        with open(os.path.join('traffic_light_detection_module', 'config.json')) as config_file:
            config = json.load(config_file)
        model = YOLO(config)
        model.model.load_weights(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                              'traffic_light_detection_module',
                                              'checkpoints', config['model']['saved_model_name']))
        print("HEY; WEIGHTS LOADED")
