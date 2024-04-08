import math
import cv2
from PIL import Image
from IPython.display import display

from src.aruco_marker import ArucoMarker
from src.detected_object import DetectedObject


def draw_bboxes(img: cv2.typing.MatLike, detected_objects: [DetectedObject], bbox_color: cv2.typing.Scalar,
                bbox_center_color: cv2.typing.Scalar, line_width: int, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    for detected_object in detected_objects:
        x1, y1, x2, y2 = detected_object.bbox

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, line_width)
        cv2.circle(img, detected_object.get_center(), 5, bbox_center_color, -1)

        result_name = detected_object.class_name
        cv2.putText(img, result_name, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, bbox_color, line_width)
    return img


def draw_circle_around_detected_object(img: cv2.typing.MatLike, detected_object: DetectedObject, marker: ArucoMarker,
                                       radius_in_centimeters: int, circle_color: cv2.typing.Scalar, line_width: int,
                                       override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    radius_in_pixels = int(radius_in_centimeters * marker.get_pixels_per_centimeter())
    cv2.circle(img, detected_object.get_center(), radius_in_pixels, circle_color, line_width)

    return img


def get_objects_around_detected_object(detected_objects: [DetectedObject], center_object: DetectedObject, marker: ArucoMarker,
                                       radius_in_centimeters: int) -> [DetectedObject]:
    object_around = []

    radius_in_pixels = int(radius_in_centimeters * marker.get_pixels_per_centimeter())
    center_point = center_object.get_center()

    for detected_object in detected_objects:
        x1, y1, x2, y2 = detected_object.bbox
        distance = math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)
        if distance < radius_in_pixels and detected_object.bbox not in center_object.bbox:
            object_around.append(detected_object)

    return object_around


def display_image(img: cv2.typing.MatLike) -> None:
    preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(preview))
