import os
import cv2 as cv
import numpy as np
from typing import Tuple, List, Union

from src.detection_helpers import make_bbox_larger, clip_detections
from src.detectors.base_detector import BaseDetector, get_field


class FaceDetectorCV2(BaseDetector):
    """Class that implements face detection using a caffe model from OpenCV"""

    def __init__(self, config):
        self.face_detector = None
        self.model_path_protobuf = os.getcwd() + get_field(config, "model_path_protobuf")
        self.model_path_caffe = os.getcwd() + get_field(config, "model_path_caffe")
        self.confidence_thresh = get_field(config, "confidence_thresh")
        self.enlarge_face_percentage = get_field(config, "enlarge_face_percentage")
        self.input_size = get_field(config, "input_size", eval_field=True)
        self.mean_normalization = get_field(config, "mean_normalization", eval_field=True)
        self.scale_factor = get_field(config, "scale_factor")
        self.load()

    def load(self):
        """ Loads the model for detection."""
        self.face_detector = cv.dnn.readNetFromCaffe(self.model_path_protobuf, self.model_path_caffe)

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
        h, w = image.shape[:2]
        # Preprocess the image by resizing it and converting it to a blob
        resized_image = cv.resize(image, self.input_size)
        blob = cv.dnn.blobFromImage(resized_image, self.scale_factor, self.input_size,
                                    self.mean_normalization)

        # Feed the blob as input to the DNN Face Detector model
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out detections with lower confidence
            if confidence > self.confidence_thresh:
                # Get the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = make_bbox_larger(box, self.enlarge_face_percentage)
                faces.append((int(startX), int(startY), int(endX - startX), int(endY - startY)))

        if faces:
            # Making sure all the face detections lie within the image
            faces = clip_detections(faces, image.shape)

        return faces
