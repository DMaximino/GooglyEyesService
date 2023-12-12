import os

# Face detection parameters
FACE_DET_CONFIDENCE_THRESH = 0.4
ENLARGE_FACE_DET_PERCENTAGE = 0.15
FACE_DET_INPUT_SIZE = (300, 300)
FACE_DET_MEAN_NORMALIZATION = (104.0, 177.0, 123.0)
FACE_DET_SCALE_FACTOR = 1.0

# Face detection model path
directory_path = os.getcwd() + '/models/'
FACE_DET_MODEL_PATH_PROTOBUF = directory_path + "deploy.prototxt"
FACE_DET_MODEL_PATH_CAFFE = directory_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Facial landmark detector path
LANDMARK_DET_MODEL_PATH = os.getcwd() + '/models/lbfmodel.yaml'
