import random
import logging
import cv2 as cv
import numpy as np
from typing import List, Tuple

import setup_logger
from constants import *


# Auxiliary functions
def make_bbox_larger(coordinates: Tuple[int, int, int, int], percentage: float) -> Tuple[float, float, float, float]:
    """ Makes a bounding box larger. It receives as input a bounding box in the format (xmin, ymin, xmax, ymax)
    and makes it larger by a percentage factor given by the input.

    Args:
        coordinates: Tuple representing bounding box coordinates in the format (xmin, ymin, xmax, ymax).
        percentage: The percentage by which to enlarge the bounding box.

    Returns: The new bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """

    xmin, ymin, xmax, ymax = [float(i) for i in coordinates]
    width = xmax - xmin
    height = ymax - ymin
    xmin -= percentage * float(width)
    xmax += percentage * float(width)
    ymin -= percentage * float(height)
    ymax += percentage * float(height)

    return xmin, ymin, xmax, ymax


def clip_detections(detections: List[Tuple[int, int, int, int]], image_shape) -> List[Tuple[int, int, int, int]]:
    """ Clips the detections to the boundaries of the image. In the case that a detection is completely out of the image
    it removes it completely from the list.

    Args:
        detections: List made out of tuples representing the coordinates of a detection in the format:
                    (x, y, width, height).
        image_shape: Tuple with the shape of the image in the format (h, w, c).

    Returns: The list of detections clipped to the boundary of the image in the same format as the input.
    """

    clipped_detections: List[Tuple] = []
    for i in range(len(detections)):
        x, y, w, h = detections[i]
        x_end = x + w
        y_end = y + h
        img_height, img_width, _ = image_shape

        # The detection is completely outside or the image
        if x > img_width or y > img_height or x_end < 0 or y_end < 0:
            continue

        # If one of the sides of the detection is outside the image it clips it to its boundaries.
        clipped_detection = list(detections[i])
        if x < 0:
            clipped_detection[0] = 0
            # When clipping x the width of the detection should account for the change
            # The width should be reduced by the distance between x and 0 in order for the x_end to be in the same
            # place as before. Works similarly for the height.
            clipped_detection[2] += int(x)
            w = clipped_detection[2]
        if y < 0:
            clipped_detection[1] = 0
            # When clipping y the height of the detection should account for the change
            clipped_detection[3] += int(y)
            h = clipped_detection[3]

        if x + w > img_width:
            clipped_detection[2] = int(img_width - x)
        if y + h > img_height:
            clipped_detection[3] = int(img_height - y)

        clipped_detections.append(tuple(clipped_detection))

    return clipped_detections


class Googlifier:
    """ Class responsible for generating the googly eye filter """

    def __init__(self):
        # Load face detection model
        self.face_detector = cv.dnn.readNetFromCaffe(FACE_DET_MODEL_PATH_PROTOBUF, FACE_DET_MODEL_PATH_CAFFE)

        # Load facial landmark detection model
        self.landmark_detector = cv.face.createFacemarkLBF()
        self.landmark_detector.loadModel(LANDMARK_DET_MODEL_PATH)

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
            image_numpy_array = np.frombuffer(image_byte_array, np.uint8)
            image_cv = cv.imdecode(image_numpy_array, cv.IMREAD_COLOR)
        except Exception as e:
            self.logger.info("Error converting image from bytes to numpy array: " + repr(e))
            return False, image_byte_array

        # Detect all faces in input image
        faces = self.detect_faces(image_cv)

        if faces:
            # Making sure all the face detections lie within the image
            faces = clip_detections(faces, image_cv.shape)

        if not faces:
            self.logger.info("No faces detected in the image.")
            return True, bytes(image_numpy_array)

        # Detect eyes in all detected faces
        eyes = self.detect_eyes(image_cv, faces)
        if not eyes:
            self.logger.info("No eyes detected in any detected face.")
            return True, bytes(image_numpy_array)

        # Draw googly eyes on image
        image_cv = Googlifier.draw_googly_eyes(eyes, image_cv)

        # Convert image from numpy array to bytes
        _, image_png = cv.imencode(".png", image_cv)

        return True, image_png.tobytes()

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

        h, w = image.shape[:2]
        # Preprocess the image by resizing it and converting it to a blob
        resized_image = cv.resize(image, FACE_DET_INPUT_SIZE)
        blob = cv.dnn.blobFromImage(resized_image, FACE_DET_SCALE_FACTOR, FACE_DET_INPUT_SIZE,
                                    FACE_DET_MEAN_NORMALIZATION)

        # Feed the blob as input to the DNN Face Detector model
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out detections with lower confidence
            if confidence > FACE_DET_CONFIDENCE_THRESH:
                # Get the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = make_bbox_larger(box, ENLARGE_FACE_DET_PERCENTAGE)
                faces.append((int(startX), int(startY), int(endX - startX), int(endY - startY)))

        return faces

    def detect_eyes(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """ Detects the eyes in each detected face.

        Args:
            image: The input image as a numpy array in the RGB colorspace.
            faces: List made out of the coordinates of each face in the format: (x, y, width, height).

        Returns: A list made out of tuples representing the coordinates of an eye in the format (x, y, width, height).
                 If no eyes are detected an empty list is returned.
        """
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, landmarks = self.landmark_detector.fit(gray_image, np.array(faces))
        eyes = []
        for landmark in landmarks:
            left_eye = cv.boundingRect(landmark[0][36:42])
            right_eye = cv.boundingRect(landmark[0][42:48])

            eyes.append(tuple(left_eye))
            eyes.append(tuple(right_eye))
        return eyes

    @staticmethod
    def draw_googly_eyes(eyes: List[Tuple[int, int, int, int]], image: np.ndarray) -> np.ndarray:
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
