from typing import Any

from fastapi.testclient import TestClient
from src.api.api import app
import base64
import os


# Create a TestClient instance for testing
client = TestClient(app)


def call_api_image_base64(filename: str) -> Any:
    """ Opens image from file with filename as input, converts it to base64 and makes a request to the API.

    Args:
        filename: String with the path to the image file.

    Returns: The response returned by the API.
    """
    with open(filename, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read())
    input_dict = {"base64_str": image_base64.decode('utf-8')}

    return client.post("/googlify/", json=input_dict)


def test_googlify_endpoint():
    """ Tests the API endpoint "googlify" with different scenarios """

    # Test with successful image with faces on it
    filename = os.getcwd() + "/tests/test_data/people_test_image.jpg"
    response = call_api_image_base64(filename)

    assert response.status_code == 200

    # Test with successful image without faces on it
    filename = os.getcwd() + "/tests/test_data/no_faces_test_image.jpg"
    response = call_api_image_base64(filename)

    assert response.status_code == 200

    # Test invalid input (content type is not an image)
    response = client.post("/googlify/", json={"base64_str": "string"})
    assert response.status_code == 400
