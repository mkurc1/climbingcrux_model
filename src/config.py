import os
from pathlib import Path
import cv2


BASE_DIR = Path(__file__).resolve().parent.parent

MARKER_PERIMETER_IN_CM = 4 * 7
MARKER_ARUCO_DICT = cv2.aruco.DICT_5X5_50

YOLO_CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
YOLO_BASE_MODEL = 'yolov9c.pt'
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'notebooks', 'runs', 'detect', 'train3', 'weights', 'best.pt')

MARKER_BBOX_COLOR = (0, 255, 0)
TXT_COLOR = (0, 0, 0)
TXT_BG_COLOR = (255, 255, 255)
BBOX_COLOR = (0, 255, 0)
BBOX_CENTER_COLOR = (0, 0, 255)

LINE_WIDTH = 2
