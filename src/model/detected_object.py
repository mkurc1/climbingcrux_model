from dataclasses import dataclass
import numpy as np

from src.model.point import Point


@dataclass
class DetectedObject:
    class_name: str
    bbox: np.ndarray
    center: Point

    def __eq__(self, other):
        return self.class_name == other.class_name and np.array_equal(self.bbox, other.bbox)
