import os
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np

from src.detectors.base_detector import BaseDetector, get_field


class EyesDetectorCV2(BaseDetector):
    """Class that implements eye detection using the LBF landmark face detector from OpenCV"""

    def __init__(self, config):
        self.landmark_detector = None
        self.model_path = os.getcwd() + get_field(config, "model_path")
        self.load()

    def load(self):
        """ Loads the model for detection."""
        self.landmark_detector = cv.face.createFacemarkLBF()
        self.landmark_detector.loadModel(self.model_path)

    def detect(self, image: np.ndarray, roi: Union[List[Tuple[int, int, int, int]], None] = None) \
            -> List[Tuple[int, int, int, int]]:
        """ Performs detection on the image taking into account the regions of interest (roi).
        If the roi is None then the detection is performed in the whole image.

        Args:
            image: Image to use for detection.
            roi: List of coordinates of regions of interest where to look for the detections.

        Returns:
            List of coordinates with detections.
        """
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, landmarks = self.landmark_detector.fit(gray_image, np.array(roi))
        eyes = []
        for landmark in landmarks:
            left_eye = cv.boundingRect(landmark[0][36:42])
            right_eye = cv.boundingRect(landmark[0][42:48])

            eyes.append(tuple(left_eye))
            eyes.append(tuple(right_eye))
        return eyes
