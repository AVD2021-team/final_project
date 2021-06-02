import sys
import os
import json
from typing import List, Dict
from enum import Enum

# Script level imports
from traffic_light_detection_module.yolo import YOLO
from traffic_light_detection_module.preprocessing import preprocess_image
from traffic_light_detection_module.postprocessing import draw_boxes, BoundBox
from data_visualization import Sensor


class TrafficLightState(Enum):
    GO = 0
    STOP = 1
    NO_TL = -1


class TrafficLightDetector(YOLO):
    # Minimum threshold to refuse false positives
    MIN_TH = 0.1

    # Minimum number of frames before change state
    MIN_STOP_FRAMES = 5     # Min frames to choice STOP signal 
    MIN_GO_FRAMES = 10      # Min frames to choice GO signal
    MIN_NOTL_FRAMES = 10    # Min frames to choice NO_TL signal
    MIN_INDECISION_FRAMES = 30  # Min frames to understand the state predicted is not reliable 

    __slots__ = 'config', 'labels', '_state_counter', '_indecision_counter', '_state', '_new_state'

    def __init__(self):
        with open(os.path.join('traffic_light_detection_module', 'config.json')) as config_file:
            self.config = json.load(config_file)
        self.labels = self.config['model']['classes']
        super().__init__(self.config)
        self.model.load_weights(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                             'traffic_light_detection_module',
                                             'checkpoints', self.config['model']['saved_model_name']))
        self._state_counter = 0
        self._indecision_counter = 0
        self._state = None
        self._new_state = None

    def get_state(self):
        return self._state

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

    def update_state(self, boxes: Dict[Sensor, List[BoundBox]]):
        """
        Returns the state of the nearest traffic light. The traffic lights are detected in boxes.
        Proximity and is inferred with a heuristic based on box area. if a red light is detected 
        in one of the rooms, this signal is given priority. Different num frames have been used according to the signal type to prioritize the signal type. 

        Args:
            boxes (Dict[Sensor, List[BoundBox]]): Dict with key the sensor and 
                                                  value the list of boxes detected in imege sensors

        Returns:
            Traffic Light State detected.
        """
        medium_state = self._light_state(boxes[Sensor.MediumFOVCameraRGB])
        large_state = self._light_state(boxes[Sensor.LargeFOVCameraRGB])
        right_state = self._light_state(boxes[Sensor.RightLargeFOVCameraRGB])

        if (medium_state[0] == TrafficLightState.STOP or
                large_state[0] == TrafficLightState.STOP or
                right_state[0] == TrafficLightState.STOP):
            new_state = TrafficLightState.STOP, min(medium_state[1], large_state[1])
        elif medium_state[0] == TrafficLightState.NO_TL:
            if large_state[0] == TrafficLightState.NO_TL:
                new_state = right_state
            else:
                new_state = large_state
        else:
            new_state = medium_state

        if self._state is None:
            self._state = (TrafficLightState.NO_TL, 1.0)
            self._new_state = (TrafficLightState.NO_TL, 1.0)

        if new_state[0] != self._new_state[0]:
            self._indecision_counter += 1
            self._new_state = new_state
            self._state_counter = 0
        elif new_state[0] == self._new_state[0]:
            self._state_counter += 1
            self._new_state = new_state

        if self._indecision_counter >= self.MIN_INDECISION_FRAMES:
            print(f"Cannot decide TL state. Defaulting to NO_TL...")
            self._state = (TrafficLightState.NO_TL, 1.0)
            self._indecision_counter = 0
        elif ((self._new_state[0] == TrafficLightState.NO_TL and self._state_counter >= self.MIN_NOTL_FRAMES) or
              (self._new_state[0] == TrafficLightState.GO and self._state_counter >= self.MIN_GO_FRAMES) or
              (self._new_state[0] == TrafficLightState.STOP and self._state_counter >= self.MIN_STOP_FRAMES)):
            self._state = self._new_state
            self._state_counter = 0
            self._indecision_counter = 0

        return self.get_state()

    @staticmethod
    def _light_state(boxes: List[BoundBox]):
        """
        filters all the detected boxes greater than MIN TH and 
        it returns the predicted label with the relative score

        Args:
            boxes (List[BoundBox]): list of Boundi box predicted
        """
        boxes_ = list(filter(lambda b: b.get_score() > TrafficLightDetector.MIN_TH, boxes))

        if len(boxes_) == 0:
            return TrafficLightState.NO_TL, 1.0

        # Heuristic
        box = sorted(boxes_, key=lambda b: b.get_area(), reverse=True)[0]

        # if box.get_label() == TrafficLightState.STOP.value:
        #    return TrafficLightState.STOP, box.get_score()

        if box.get_label() == TrafficLightState.GO.value:
            return TrafficLightState.GO, box.get_score()

        return TrafficLightState.STOP, box.get_score()
