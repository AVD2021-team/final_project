import sys
import os
from enum import Enum
import cv2

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla.image_converter as image_converter


class Sensor(Enum):
    LargeFOVCameraRGB = "LargeFOVCameraRGB"
    NarrowFOVCameraRGB = "NarrowFOVCameraRGB"
    MediumFOVCameraRGB = "MediumFOVCameraRGB"
    RightLargeFOVCameraRGB = "RightLargeFOVCameraRGB"
    DepthCamera = "DepthCamera"


def visualize_sensor_data(sensor_data, sensor, showing_dims=None):
    """
    Visualizes the output of sensor
    sensor_data is the [1] output of carla.client.make_carla_client.read_data()
    sensor is one of Sensor Enum
    showing_dims(width, height) is the shape at which the image acquired by the sensor will be visualized
    """
    image = get_sensor_output(sensor_data, sensor)

    if image is not None:
        if showing_dims is not None:
            image = cv2.resize(image, showing_dims)
        cv2.imshow(f"{sensor.value}", image)
        cv2.waitKey(1)


def get_sensor_output(sensor_data, sensor):
    """
    Visualizes the output of sensor
    sensor_data is the [1] output of carla.client.make_carla_client.read_data()
    sensor is one of Sensor Enum
    showing_dims(width, height) is the shape at which the image acquired by the sensor will be visualized
    """
    if sensor_data.get(sensor.value, None) is not None:
        image = None

        if sensor in (Sensor.LargeFOVCameraRGB, Sensor.MediumFOVCameraRGB, Sensor.NarrowFOVCameraRGB, Sensor.RightLargeFOVCameraRGB, ):
            # Camera RGB data
            image = image_converter.to_bgra_array(sensor_data[sensor.value])
        elif sensor in (Sensor.DepthCamera,):
            image = image_converter.depth_to_array(sensor_data[sensor.value])
        else:
            raise RuntimeError(f"get_sensor_output not implemented for {sensor.value}")

    return image

