import os
from pathlib import Path
import cv2
from dotenv import load_dotenv

from src.model.color import Color

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_FILE_PATH = os.path.join(BASE_DIR, '.env')

load_dotenv(dotenv_path=ENV_FILE_PATH)

MARKER_PERIMETER_IN_CM = 4 * 7
MARKER_ARUCO_DICT = cv2.aruco.DICT_5X5_50

YOLO_CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
YOLO_BASE_MODEL = 'yolov9c.pt'
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'model', os.getenv('YOLO_MODEL_PATH'))

# Chose Apple Metal Performance Shaders (MPS) if available, otherwise use CPU
# YOLO_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
# Use temporary CPU for now. There is some issue with MPS
YOLO_DEVICE = 'cpu'

MARKER_BBOX_COLOR = Color.green()
TXT_COLOR = Color.black()
TXT_BG_COLOR = Color.white()
BBOX_COLOR = Color.green()
BBOX_CENTER_COLOR = Color.red()

PROBLEM_STEP_BBOX_COLOR = Color.red()

CLIMBER_HEIGHT_IN_CM = 170
STEP_RADIUS_IN_CM = 70
STARTING_STEPS_MAX_DISTANCE_FROM_GROUND_IN_CM = 40

LINE_WIDTH = 2

MAXIMUM_FILE_SIZE = 1024 * 1024 * 4  # 4MB
ACCEPTED_MIME_TYPES = ["image/png", "image/jpeg", "image/jpg"]
