import cv2
from PIL import Image
from IPython.display import display

from src.aruco_marker import ArucoMarker
from src.model.detected_object import DetectedObject


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

    return draw_circle_around_point(img, detected_object.get_center(), marker, radius_in_centimeters, circle_color,
                                    line_width, override)


def draw_circle_around_point(img: cv2.typing.MatLike, point: cv2.typing.Point, marker: ArucoMarker,
                                       radius_in_centimeters: int, circle_color: cv2.typing.Scalar, line_width: int,
                                       override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    radius_in_pixels = int(radius_in_centimeters * marker.get_pixels_per_centimeter())
    cv2.circle(img, point, radius_in_pixels, circle_color, line_width)

    return img


def draw_line(img: cv2.typing.MatLike, start_point: cv2.typing.Point, end_point: cv2.typing.Point,
              color: cv2.typing.Scalar, thickness: int, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.line(img, start_point, end_point, color, thickness)

    return img


def display_image(img: cv2.typing.MatLike) -> None:
    preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(preview))
