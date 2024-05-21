import cv2
from PIL import Image
from IPython.display import display

from src import config
from src.model.climber import Climber
from src.model.detected_object import DetectedObject
from src.model.body_part import BodyPart
from src.model.color import Color
from src.model.point import Point


def draw_bboxes(img: cv2.typing.MatLike, detected_objects: [DetectedObject], bbox_color: Color,
                bbox_center_color: Color, line_width: int, draw_labels: bool = True,
                draw_centers: bool = True, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    for detected_object in detected_objects:
        x1, y1, x2, y2 = detected_object.bbox

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color.bgr(), line_width)

        if draw_centers:
            cv2.circle(img, detected_object.center.to_tuple(), 5, bbox_center_color.bgr(), -1)

        if draw_labels:
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


def draw_line(img: cv2.typing.MatLike, start_point: Point, end_point: Point,
              color: Color, line_width: int, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.line(img, start_point.to_tuple(), end_point.to_tuple(), color.bgr(), line_width)

    return img


def draw_rectangle(img: cv2.typing.MatLike, start_point: Point, end_point: Point, color: Color, line_width: int,
                   override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.rectangle(img, start_point.to_tuple(), end_point.to_tuple(), color.bgr(), line_width)

    return img


def draw_body_part(img: cv2.typing.MatLike, body_part: BodyPart, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    cv2.line(img, body_part.start.to_tuple(), body_part.end.to_tuple(),
             body_part.color.bgr(), body_part.thickness)

    return img


def draw_climber(img: cv2.typing.MatLike, climber: Climber, body: bool = False,
                 draw_labels: bool = False, draw_centers: bool = False, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    for body_part in [climber.head, climber.neck, climber.trunk, climber.left_shoulder, climber.right_shoulder,
                      climber.left_leg, climber.right_leg, climber.left_arm, climber.right_arm]:
        if body:
            draw_body_part(
                img,
                body_part,
            )

        if body_part.detected_object is not None:
            draw_bboxes(
                img=img,
                detected_objects=[body_part.detected_object],
                bbox_color=config.PROBLEM_STEP_BBOX_COLOR,
                bbox_center_color=config.BBOX_CENTER_COLOR,
                line_width=config.LINE_WIDTH,
                draw_labels=draw_labels,
                draw_centers=draw_centers,
            )

    return img


def display_image(img: cv2.typing.MatLike) -> None:
    preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(preview))
