import numpy as np

from src.aruco_marker import ArucoMarker
from src.model.body_part import BodyPart
from src.model.climber import Climber
from src.model.color import Color
from src.model.detected_object import DetectedObject
from src import objects_detector, config
from src.model.point import Point


class ClimberStartPosition:
    def __init__(self, img_width: int, img_height: int, marker: ArucoMarker,
                 detected_objects: [DetectedObject]):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__marker = marker
        self.__detected_objects = detected_objects

    def prepare(self, climber_height_in_cm: int,
                starting_steps_max_distance_from_ground_in_cm: int) -> Climber:
        climber_height_in_px = self.__marker.convert_cm_to_px(climber_height_in_cm)

        climber = Climber(climber_height_in_px)

        bottom_objects = self.__get_bottom_objects_fit_as_steps(
            starting_steps_max_distance_from_ground_in_cm
        )

        starting_step_1_id = np.random.choice(len(bottom_objects), 1, replace=False)[0]
        starting_step_1 = bottom_objects[starting_step_1_id]

        starting_step_2 = self.__find_hold_in_circle(
            detected_objects=bottom_objects,
            point=starting_step_1.center,
            radius=self.__marker.convert_cm_to_px(config.STEP_RADIUS_IN_CM),
            exclude_detected_objects=[starting_step_1]
        )

        if starting_step_1.center.x < starting_step_2.center.x:
            starting_left_step = starting_step_1
            starting_right_step = starting_step_2
        else:
            starting_left_step = starting_step_2
            starting_right_step = starting_step_1

        lower_starting_step = starting_step_1 \
            if starting_step_1.center.y > starting_step_2.center.y \
            else starting_step_2

        body_center = int((starting_left_step.center.x +
                           starting_right_step.center.x) / 2)

        top_head_point = Point(
            body_center,
            int(lower_starting_step.center.y - climber.body_proportion.height))
        bottom_head_point = Point(
            body_center,
            int(lower_starting_step.center.y - climber.body_proportion.height +
                climber.body_proportion.head))

        climber.head = BodyPart(
            start=top_head_point,
            end=bottom_head_point,
            color=Color(255, 255, 0),  # yellow color
            thickness=30
        )

        top_neck_point = Point(top_head_point.x, int(bottom_head_point.y))
        bottom_neck_point = Point(
            bottom_head_point.x,
            int(bottom_head_point.y + climber.body_proportion.neck)
        )

        climber.neck = BodyPart(
            start=top_neck_point,
            end=bottom_neck_point,
            color=Color(0, 125, 255),  # orange color
            thickness=10
        )

        top_trunk_point = Point(top_head_point.x, int(bottom_neck_point.y))
        bottom_trunk_point = Point(
            bottom_head_point.x,
            int(bottom_neck_point.y + climber.body_proportion.trunk)
        )

        climber.trunk = BodyPart(
            start=top_trunk_point,
            end=bottom_trunk_point,
            color=Color.green(),  # green color
            thickness=30
        )

        start_left_shoulder_point = Point(
            bottom_neck_point.x - int(climber.body_proportion.shoulder),
            bottom_neck_point.y
        )
        end_left_shoulder_point = Point(bottom_neck_point.x, bottom_neck_point.y)

        climber.left_shoulder = BodyPart(
            start=start_left_shoulder_point,
            end=end_left_shoulder_point,
            color=Color.blue(),
            thickness=10
        )

        start_right_shoulder_point = Point(
            bottom_neck_point.x, bottom_neck_point.y
        )
        end_right_shoulder_point = Point(
            bottom_neck_point.x + int(climber.body_proportion.shoulder),
            bottom_neck_point.y
        )

        climber.right_shoulder = BodyPart(
            start=start_right_shoulder_point,
            end=end_right_shoulder_point,
            color=Color.blue(),
            thickness=10
        )

        start_left_leg_point = Point(
            bottom_trunk_point.x, bottom_trunk_point.y
        )
        end_left_leg_point = starting_left_step.center

        climber.left_leg = BodyPart(
            start=start_left_leg_point,
            end=end_left_leg_point,
            color=Color.blue(),
            thickness=10,
            detected_object=starting_left_step
        )

        start_right_leg_point = Point(
            bottom_trunk_point.x, bottom_trunk_point.y
        )
        end_right_leg_point = starting_right_step.center

        climber.right_leg = BodyPart(
            start=start_right_leg_point,
            end=end_right_leg_point,
            color=Color.blue(),
            thickness=10,
            detected_object=starting_right_step
        )

        climber.left_hand = self.__find_hold_for_left_hand(
            climber, body_center)
        climber.right_hand = self.__find_hold_for_right_hand(
            climber, body_center)

        return climber

    def __get_bottom_objects_fit_as_steps(self, max_distance_from_ground_in_cm: int) -> [DetectedObject]:
        max_distance_from_ground_in_px = (
            self.__marker.convert_cm_to_px(max_distance_from_ground_in_cm))

        # 40 cm of bottom boxes from image but exclude from left and right 15%
        return [obj for obj in self.__detected_objects if
                obj.bbox[3] > self.__img_height - max_distance_from_ground_in_px and
                obj.bbox[0] > 0.15 * self.__img_width and
                obj.bbox[2] < 0.85 * self.__img_width
                ]

    def __find_hold_for_left_hand(self, climber: Climber,
                                  body_center: int) -> BodyPart:
        # get holds available for left hand
        holds = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=climber.left_shoulder.start,
            radius=int(climber.body_proportion.arm)
        )

        # exclude holds that are on the right side of
        # the body_center
        holds = [hold for hold in holds
                 if hold.center.x < body_center]

        random_left_hand_hold_id = np.random.choice(len(holds), 1, replace=False)[0]
        random_left_hand_hold = holds[random_left_hand_hold_id]

        return BodyPart(
            start=climber.left_shoulder.start,
            end=random_left_hand_hold.center,
            color=Color.red(),
            thickness=10,
            detected_object=random_left_hand_hold
        )

    def __find_hold_for_right_hand(self, climber: Climber,
                                   body_center: int) -> BodyPart:
        # get holds available for right hand
        holds = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=climber.right_shoulder.end,
            radius=int(climber.body_proportion.arm)
        )

        # exclude holds that are on the left side of
        # the body_center
        holds = [hold for hold in holds
                 if hold.center.x > body_center]

        random_right_hand_hold_id = np.random.choice(len(holds), 1, replace=False)[0]
        random_right_hand_hold = holds[random_right_hand_hold_id]

        return BodyPart(
            start=climber.right_shoulder.end,
            end=random_right_hand_hold.center,
            color=Color.red(),
            thickness=10,
            detected_object=random_right_hand_hold
        )

    def __find_hold_in_circle(self, detected_objects: [DetectedObject],
                              point: Point, radius: int,
                              exclude_detected_objects: [DetectedObject] = ()
                              ) -> DetectedObject:
        holds_in_circle = objects_detector.get_objects_around_point(
            detected_objects=detected_objects,
            point=point,
            radius=radius,
            exclude_detected_objects=exclude_detected_objects
        )

        object_detect_id = np.random.choice(len(holds_in_circle), 1, replace=False)[0]
        return holds_in_circle[object_detect_id]
