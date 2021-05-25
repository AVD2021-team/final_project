import sys
import os
import json
from typing import List
from enum import Enum

# Script level imports
from traffic_light_detection_module.yolo import YOLO
from traffic_light_detection_module.preprocessing import preprocess_image
from traffic_light_detection_module.postprocessing import draw_boxes, BoundBox


class TrafficLightState(Enum):
    GO = 0
    STOP = 1
    NO_TL = -1


class TrafficLightDetector(YOLO):

    __slots__ = 'config', 'labels'

    def __init__(self):
        with open(os.path.join('traffic_light_detection_module', 'config.json')) as config_file:
            self.config = json.load(config_file)
        self.labels = self.config['model']['classes']
        super().__init__(self.config)
        self.model.load_weights(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                             'traffic_light_detection_module',
                                             'checkpoints', self.config['model']['saved_model_name']))

    def predict_image(self, image):
        """
        Predicts the bounding boxes of traffic lights from a cv2 image
        Returns a list of BoundBox objects.
        """
        image = preprocess_image(image, self.image_h, self.image_w)
        boxes = super().predict_image(image)
        return boxes

    def draw_boxes(self, image, boxes):
        """
        Draws the detected traffic lights' boxes on a cv2 image.
        Returns the new image.
        """
        return draw_boxes(image, boxes, self.labels)

    @staticmethod
    def light_state(boxes: List[BoundBox]):
        """
        Returns the state of the nearest traffic light. The traffic lights are detected in boxes.
        Proximity is inferred with a heuristic based on box area.
        """
        if len(boxes) == 0:
            return TrafficLightState.NO_TL

        box = sorted(boxes, key=lambda b: b.get_area())[0]

        if box.get_label() == TrafficLightState.STOP.value:
            return TrafficLightState.STOP

        return TrafficLightState.GO
