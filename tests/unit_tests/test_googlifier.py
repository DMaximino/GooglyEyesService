import cv2
import os
from src.googlifier import Googlifier, make_bbox_larger, clip_detections


def test_clip_detections():
    detections = [(-1, -2, 5, 6), (1, 2, 4, 4), (-10, -20, -1, -2), (10, 20, 1, 2)]
    image_shape = (4, 4, 3)
    new_detections = clip_detections(detections, image_shape)

    assert new_detections == [(0, 0, 4, 4), (1, 2, 3, 2)]


def test_make_bbox_larger():
    coordinates = (5, 5, 10, 10)
    new_coordinates = make_bbox_larger(coordinates, 1)
    assert new_coordinates == (0, 0, 15, 15)

    new_coordinates = make_bbox_larger(coordinates, 0.5)
    assert new_coordinates == (2.5, 2.5, 12.5, 12.5)


def test_detect_faces():
    googly = Googlifier()

    # Detect faces in image with faces
    filename = os.getcwd() + "/tests/test_data/people_test_image.jpg"
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    faces = googly.detect_faces(image)
    assert len(faces) > 0

    # Detect faces in image without faces
    filename = os.getcwd() + "/tests/test_data/no_faces_test_image.jpg"
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    faces = googly.detect_faces(image)
    assert faces == []

    # Call detect faces with an input different from a numpy array
    faces = googly.detect_faces([1, 2, 3])
    assert faces == []


def test_detect_eyes():
    googly = Googlifier()

    # Detect eyes in faces
    filename = os.getcwd() + "/tests/test_data/people_test_image.jpg"
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    faces = googly.detect_faces(image)
    eyes = googly.detect_eyes(image, faces)

    assert len(eyes) > 0
