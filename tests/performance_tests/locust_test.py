from locust import task, between, FastHttpUser
import base64
import os


class PerformanceTests(FastHttpUser):
    wait_time = between(1, 3)

    @task(1)
    def test_tf_predict(self):
        """ Makes a request to the API endpoint "googlify" using a test image"""

        filename = os.getcwd() + "/tests/test_data/people_test_image.jpg"
        with open(filename, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read())
        input_dict = {"base64_str": image_base64.decode('utf-8')}

        _ = self.client.post("/googlify/", json=input_dict)
