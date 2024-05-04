from typing import Optional

from src.model.color import Color
from src.model.detected_object import DetectedObject


class BodyPart:
    def __init__(self, start: tuple[int, int], end: tuple[int, int],
                 color: Color, thickness: int = 10,
                 detected_object: Optional[DetectedObject] = None):
        self.start = start
        self.end = end
        self.color = color
        self.thickness = thickness
        self.detected_object = detected_object
