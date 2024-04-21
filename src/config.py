import os
from pathlib import Path
import cv2
from dotenv import load_dotenv

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

MARKER_BBOX_COLOR = (0, 255, 0)  # Green
TXT_COLOR = (0, 0, 0)  # Black
TXT_BG_COLOR = (255, 255, 255)  # White
BBOX_COLOR = (0, 255, 0)  # Green
BBOX_CENTER_COLOR = (0, 0, 255)  # Red

PROBLEM_STEP_BBOX_COLOR = (0, 0, 255)  # Red

CLIMBER_HEIGHT_IN_CM = 180
STEP_RADIUS_IN_CM = 80

LINE_WIDTH = 2
