import math
import cv2
from src.model.detected_object import DetectedObject
from src import config
from src.aruco_marker import ArucoMarker
from ultralytics import YOLO
import numpy as np


def detect(img: cv2.typing.MatLike, conf: float = 0.85, imgsz: int = 1216) -> [DetectedObject]:
    model = YOLO(config.YOLO_MODEL_PATH)

    results = model(
        img,
        device=config.YOLO_DEVICE,
        conf=conf,
        imgsz=imgsz,
    )

    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
    classes = np.array(result.boxes.cls.cpu(), dtype=int)
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2

    detected_objects = []
    for bbox, class_id, center in zip(bboxes, classes, centers):
        detected_objects.append(DetectedObject(result.names[class_id], bbox, center))

    return detected_objects


def get_objects_around_detected_object(detected_objects: [DetectedObject], center_object: DetectedObject,
                                       marker: ArucoMarker, radius_in_centimeters: int) -> [DetectedObject]:
    object_around = []

    radius_in_pixels = int(radius_in_centimeters * marker.get_pixels_per_centimeter())
    center_point = center_object.get_center()

    for detected_object in detected_objects:
        x1, y1, x2, y2 = detected_object.bbox
        distance = math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)
        if distance < radius_in_pixels and detected_object.bbox not in center_object.bbox:
            object_around.append(detected_object)

    return object_around
