import math
import cv2
from ultralytics import YOLO
import numpy as np

from src import config
from src.model.detected_object import DetectedObject
from src.model.point import Point


def detect(img: cv2.typing.MatLike, conf: float = 0.85,
           imgsz: int = 1216) -> [DetectedObject]:
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
        detected_objects.append(DetectedObject(
            class_name=result.names[class_id],
            bbox=bbox,
            center=Point(
                x=int(round(center[0])),
                y=int(round(center[1]))
            )
        ))

    return detected_objects


def get_objects_around_point(detected_objects: [DetectedObject],
                             point: Point, radius: int,
                             exclude_detected_objects: [DetectedObject] = ()
                             ) -> [DetectedObject]:
    objects_around = []

    for detected_object in detected_objects:
        x1, y1, x2, y2 = detected_object.bbox
        distance = math.sqrt((x1 - point.x) ** 2 + (y1 - point.y) ** 2)
        if distance < radius:
            objects_around.append(detected_object)

    for exclude_detected_object in exclude_detected_objects:
        if exclude_detected_object in objects_around:
            objects_around.remove(exclude_detected_object)

    return objects_around


def get_distance_between_objects(object1: DetectedObject,
                                 object2: DetectedObject,
                                 ) -> int:
    c1 = object1.center
    c2 = object2.center

    distance = math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    return int(round(distance))
