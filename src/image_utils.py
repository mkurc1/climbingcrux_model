import cv2
from PIL import Image
from IPython.display import display
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


def display_image(img: cv2.typing.MatLike) -> None:
    preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(preview))
