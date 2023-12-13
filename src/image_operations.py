from typing import List, Tuple
import cv2 as cv
import numpy as np
import random


def draw_googly_eyes_on_image(eyes: List[Tuple[int, int, int, int]], image: np.ndarray) -> np.ndarray:
    """ Draws the googly eyes on top of the detected eyes in the image. The googly eyes are drawn with a fixed size
    in regard to the size of the detected bounding box and the pupil is drawn with a random position and size within
    the eye.

    Args:
        eyes: A list made out of tuples representing the coordinates of an eye in the format (x, y, width, height).
        image: The input image as a numpy array in the RGB colorspace.

    Returns: An image with googly eyes drawn on top of the detected eyes as a numpy array in the BGR colorspace.
             If eyes is an empty list the input image is returned without changes.
    """
    for eye_coordinates in eyes:
        x, y, w, h = eye_coordinates
        eye_center = (x + w // 2, y + h // 2)
        radius = int(round((w + h) * 0.75))
        image = cv.circle(image, eye_center, radius, (255, 255, 255), cv.FILLED)
        image = cv.circle(image, eye_center, radius, (0, 0, 0), 2)
        pupil_size = int(radius * random.uniform(0.45, 1))
        half_pupil_size = int(pupil_size / 2)
        pupil_center = (
            eye_center[0] + random.randint(-half_pupil_size, half_pupil_size),
            eye_center[1] + random.randint(-half_pupil_size, half_pupil_size))
        image = cv.circle(image, pupil_center, half_pupil_size, (0, 0, 0), cv.FILLED)

    return image


def convert_image_to_bytes(image_cv: np.ndarray) -> bytes:
    """ Converts an image to bytes.

    Args:
        image_cv: The input image to be converted.

    Returns: The image as bytes type.
    """
    return cv.imencode(".png", image_cv)[1].tobytes()


def convert_bytes_to_image(image_byte_array: bytes) -> np.ndarray:
    """ Converts an image as type bytes into a numpy array with BGR colorspace.

    Args:
        image_byte_array: The bytes image to be converted.

    Returns: Converted image in numpy array format.
    """
    image_numpy_array = np.frombuffer(image_byte_array, np.uint8)
    return cv.imdecode(image_numpy_array, cv.IMREAD_COLOR)
