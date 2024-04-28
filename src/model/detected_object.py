from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedObject:
    class_name: str
    bbox: np.ndarray
    center: np.ndarray

    def get_center(self) -> tuple:
        return tuple(self.center.astype(int))

    def __eq__(self, other):
        return self.class_name == other.class_name and np.array_equal(self.bbox, other.bbox)
