from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedObject:
    class_name: str
    bbox: np.ndarray
    center: np.ndarray

    def get_center(self) -> tuple:
        return tuple(self.center.astype(int))
