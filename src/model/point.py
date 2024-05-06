from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return self.x, self.y
