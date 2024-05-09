from src.model.body_part import BodyPart
from src.model.body_proportion import BodyProportion
from src.model.point import Point


class Climber:
    head: BodyPart = None
    neck: BodyPart = None
    left_shoulder: BodyPart = None
    right_shoulder: BodyPart = None
    trunk: BodyPart = None
    left_arm: BodyPart = None
    right_arm: BodyPart = None
    left_leg: BodyPart = None
    right_leg: BodyPart = None

    def __init__(self, height: int):
        self.body_proportion = BodyProportion(height)

    def get_top_left_point(self) -> Point:
        return self.head.start

    def get_top_right_point(self) -> Point:
        return self.head.end

    def get_bottom_left_point(self) -> Point:
        return self.left_leg.end

    def get_bottom_right_point(self) -> Point:
        return self.right_leg.end

    def get_lower_step_point(self) -> Point:
        if self.left_leg.detected_object.center.y > self.right_leg.detected_object.center.y:
            return self.left_leg.end
        return self.right_leg.end
