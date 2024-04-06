import cv2
import numpy as np
from PIL import Image
from IPython.display import display


def draw_bboxes(img: cv2.typing.MatLike, bboxes: np.ndarray, classes: np.ndarray, class_names: dict,
                centers: np.ndarray, bbox_color: cv2.typing.Scalar, bbox_center_color: cv2.typing.Scalar,
                line_width: int, override: bool = True) -> cv2.typing.MatLike:
    if not override:
        img = img.copy()

    for cls, bbox, center in zip(classes, bboxes, centers):
        x1, y1, x2, y2 = bbox

        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, line_width)
        cv2.circle(img, (int(center[0]), int(center[1])), 5, bbox_center_color, -1)

        result_name = class_names[cls]
        cv2.putText(img, result_name, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, bbox_color, line_width)
    return img


def display_image(img: cv2.typing.MatLike) -> None:
    preview = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
    display(Image.fromarray(preview))
