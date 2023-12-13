"""
This is an app to run the googlifier class locally and in realtime using the webcam.
"""

import cv2 as cv
import numpy as np

from googlifier import Googlifier
from constants import *

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if cap.isOpened is False:
    print('Error opening video capture.')
    exit(0)

googly = Googlifier(CONFIG_FILE_PATH)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('No captured frame.')
        break

    # Preprocess frame
    _, image_png = cv.imencode(".png", frame)
    image_byte_array = image_png.tobytes()

    # Add googly eyes
    _, frame = googly.detect_eyes_and_googlify(image_byte_array)

    # Postprocess frame
    image_numpy_array = np.frombuffer(frame, np.uint8)
    frame = cv.imdecode(image_numpy_array, cv.IMREAD_COLOR)

    # Display result
    cv.imshow('Googly eyes', frame)

    if cv.waitKey(10) == 27:
        break
