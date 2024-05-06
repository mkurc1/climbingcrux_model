import cv2
from PIL import Image
from IPython.display import display

from src.model.detected_object import DetectedObject
from src.model.body_part import BodyPart
from src.model.color import Color
from src.model.point import Point


def draw_bboxes(img: cv2.typing.MatLike, detected_objects: [DetectedObject], bbox_color: Color,
                bbox_center_color: Color, line_width: int, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    for detected_object in detected_objects:
        x1, y1, x2, y2 = detected_object.bbox

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color.bgr(), line_width)
        cv2.circle(img, detected_object.center.to_tuple(), 5, bbox_center_color.bgr(), -1)

        result_name = detected_object.class_name
        cv2.putText(img, result_name, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, bbox_color.bgr(), line_width)
    return img


def draw_circle_around_detected_object(img: cv2.typing.MatLike, detected_object: DetectedObject,
                                       radius: int, circle_color: Color, line_width: int,
                                       override: bool = True) -> cv2.typing.MatLike:
    return draw_circle_around_point(img, detected_object.center, radius, circle_color,
                                    line_width, override)


def draw_circle_around_point(img: cv2.typing.MatLike, point: Point, radius: int,
                             circle_color: Color, line_width: int,
                             override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.circle(img, point.to_tuple(), radius, circle_color.bgr(), line_width)

    return img


def draw_line(img: cv2.typing.MatLike, start_point: cv2.typing.Point, end_point: cv2.typing.Point,
              color: Color, thickness: int, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.line(img, start_point, end_point, color.bgr(), thickness)

    return img


def draw_body_part(img: cv2.typing.MatLike, body_part: BodyPart, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.line(img, body_part.start.to_tuple(), body_part.end.to_tuple(),
             body_part.color.bgr(), body_part.thickness)

    return img


def display_image(img: cv2.typing.MatLike) -> None:
    preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(preview))
