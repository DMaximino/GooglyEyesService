import yaml
import logging
import numpy as np
from typing import List, Tuple

import setup_logger
from detectors.base_detector import get_detector, get_field
from image_operations import convert_bytes_to_image, convert_image_to_bytes, draw_googly_eyes_on_image


class Googlifier:
    """ Class responsible for generating the googly eye filter """

    def __init__(self, config_file_path: str):
        # Load configurations
        with open(config_file_path, "r") as file_object:
            config = yaml.load(file_object, Loader=yaml.SafeLoader)

        face_detector_config = get_field(config, "face_detector")
        eyes_detector_config = get_field(config, "eyes_detector")

        # Load face detection model
        detector_name = get_field(face_detector_config, "model_class")
        config_dict = get_field(face_detector_config,"parameters")
        self.face_detector = get_detector(detector_name)(config_dict)

        # Load eye detection model
        detector_name = get_field(eyes_detector_config, "model_class")
        config_dict = get_field(eyes_detector_config,"parameters")
        self.eyes_detector = get_detector(detector_name)(config_dict)

        # Create the Logger
        self.logger = logging.getLogger(setup_logger.LOGGER_NAME)

    def detect_eyes_and_googlify(self, image_byte_array: bytes) -> Tuple[bool, bytes]:
        """ Detects all the faces in the input image, then for each face detect the facial landmarks.  From the facial
        landmarks it extracts the eyes coordinates and draws the googly eyes on top.

        Args:
            image_byte_array: Input image as a byte array.

        Returns:
            A tuple with: A boolean representing whether the operation was successful, it is false if the provided input
            image is corrupt in some way that does not allow to be converted to a numpy array;
            The image with the googly eyes drawn on top of the detected eyes. If no eyes are detected it simply returns
            the same image.
        """
        # Convert image from bytes to numpy array
        try:
            image = convert_bytes_to_image(image_byte_array)
        except Exception as e:
            self.logger.info("Error converting image from bytes to numpy array: " + repr(e))
            return False, image_byte_array

        # Detect all faces in input image
        faces = self.detect_faces(image)

        if not faces:
            self.logger.info("No faces detected in the image.")
            return True, image_byte_array

        # Detect eyes in all detected faces
        eyes = self.detect_eyes(image, faces)
        if not eyes:
            self.logger.info("No eyes detected in any detected face.")
            return True, image_byte_array

        # Draw googly eyes on image
        image = draw_googly_eyes_on_image(eyes, image)

        # Convert image from numpy array to bytes
        image_bytes = convert_image_to_bytes(image)

        return True, image_bytes

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """ Detects all the faces in the input image.

        Args:
            image: Input image as a numpy array in the BGR color space.

        Returns: List made out of tuples representing the coordinates of each face in the format: (x, y, width, height).
                 If no faces are detected an empty list is returned.
        """
        if not isinstance(image, np.ndarray):
            self.logger.info("Image is not a numpy array.")
            return []

        return self.face_detector.detect(image)

    def detect_eyes(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """ Detects the eyes in each detected face.

        Args:
            image: The input image as a numpy array in the RGB colorspace.
            faces: List made out of the coordinates of each face in the format: (x, y, width, height).

        Returns: A list made out of tuples representing the coordinates of an eye in the format (x, y, width, height).
                 If no eyes are detected an empty list is returned.
        """
        return self.eyes_detector.detect(image, faces)
