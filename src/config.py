import os
from pathlib import Path
import cv2


BASE_DIR = Path(__file__).resolve().parent.parent

MARKER_PERIMETER_IN_CM = 4 * 7
MARKER_ARUCO_DICT = cv2.aruco.DICT_5X5_50

YOLO_CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
