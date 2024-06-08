import copy

import numpy as np

from src.aruco_marker import ArucoMarker
from src.model.body_part import BodyPart
from src.model.climber import Climber
from src.model.color import Color
from src.model.detected_object import DetectedObject
from src import objects_detector, config
from src.model.point import Point


class RouteGenerator:
    def __init__(self, img_width: int, img_height: int, marker: ArucoMarker,
                 detected_objects: [DetectedObject]):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__marker = marker
        self.__detected_objects = detected_objects

    def generate_route(self, climber_height_in_cm: int,
                       starting_steps_max_distance_from_ground_in_cm: int) -> [Climber]:
        positions: [Climber] = [self.prepare_first_position(
            climber_height_in_cm,
            starting_steps_max_distance_from_ground_in_cm
        )]

        while not self.__is_climber_on_top(positions[-1]):
            positions.append(self.prepare_next_position(positions[-1]))

        return positions

    def prepare_next_position(self, climber: Climber) -> Climber:
        new_climber = copy.deepcopy(climber)

        lower_step_point = climber.get_lower_step_point()

        twenty_percent_of_climber_leg_height = climber.body_proportion.leg * 0.2
        forty_percent_of_climber_leg_height = climber.body_proportion.leg * 0.6

        twenty_percent_of_climber_leg_height_point = Point(
            lower_step_point.x,
            int(round(lower_step_point.y - twenty_percent_of_climber_leg_height))
        )

        forty_percent_of_climber_leg_height_point = Point(
            lower_step_point.x,
            int(round(lower_step_point.y - forty_percent_of_climber_leg_height))
        )

        rectangle_bottom_left_point = Point(climber.get_bottom_left_point().x,
                                            twenty_percent_of_climber_leg_height_point.y)
        rectangle_top_right_point = Point(climber.get_bottom_right_point().x,
                                          forty_percent_of_climber_leg_height_point.y)

        rectangle_width = rectangle_top_right_point.x - rectangle_bottom_left_point.x
        if self.__marker.convert_px_to_cm(rectangle_width) < config.STEP_RADIUS_IN_CM:
            extra_width = self.__marker.convert_cm_to_px(config.STEP_RADIUS_IN_CM) - rectangle_width
            rectangle_bottom_left_point.x -= extra_width // 2
            rectangle_top_right_point.x += extra_width // 2

        holds_for_first_foot = [hold for hold in self.__detected_objects if
                                rectangle_bottom_left_point.x <= hold.center.x <= rectangle_top_right_point.x and rectangle_bottom_left_point.y >= hold.center.y >= rectangle_top_right_point.y]

        holds_for_second_foot = [hold for hold in self.__detected_objects if
                                 twenty_percent_of_climber_leg_height_point.y >= hold.center.y >= forty_percent_of_climber_leg_height_point.y]

        self.__prepare_new_position(
            climber=new_climber,
            holds_for_first_step=holds_for_first_foot,
            holds_for_second_step=holds_for_second_foot
        )

        return new_climber

    def prepare_first_position(self, climber_height_in_cm: int,
                               starting_steps_max_distance_from_ground_in_cm: int) -> Climber:
        climber_height_in_px = self.__marker.convert_cm_to_px(climber_height_in_cm)
        climber = Climber(climber_height_in_px)

        bottom_objects = self.__get_bottom_objects_fit_as_steps(
            starting_steps_max_distance_from_ground_in_cm
        )

        self.__prepare_new_position(
            climber=climber,
            holds_for_first_step=bottom_objects,
            holds_for_second_step=bottom_objects
        )

        return climber

    def __prepare_new_position(self, climber: Climber,
                               holds_for_first_step: [DetectedObject],
                               holds_for_second_step: [DetectedObject]) -> None:
        starting_step_1_id = np.random.choice(len(holds_for_first_step), 1, replace=False)[0]
        starting_step_1 = holds_for_first_step[starting_step_1_id]

        starting_step_2 = self.__find_hold_in_circle(
            detected_objects=holds_for_second_step,
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

        body_center = int(round((starting_left_step.center.x +
                                 starting_right_step.center.x) / 2))

        top_head_point = Point(
            body_center,
            int(round(lower_starting_step.center.y - climber.body_proportion.height)))
        bottom_head_point = Point(
            body_center,
            int(round(lower_starting_step.center.y - climber.body_proportion.height) +
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
            int(round(bottom_head_point.y + climber.body_proportion.neck))
        )

        climber.neck = BodyPart(
            start=top_neck_point,
            end=bottom_neck_point,
            color=Color(0, 125, 255),  # orange color
            thickness=10
        )

        top_trunk_point = Point(top_head_point.x, int(round(bottom_neck_point.y)))
        bottom_trunk_point = Point(
            bottom_head_point.x,
            int(round(bottom_neck_point.y + climber.body_proportion.trunk))
        )

        climber.trunk = BodyPart(
            start=top_trunk_point,
            end=bottom_trunk_point,
            color=Color.green(),  # green color
            thickness=30
        )

        start_left_shoulder_point = Point(
            bottom_neck_point.x - int(round(climber.body_proportion.shoulder)),
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
            bottom_neck_point.x + int(round(climber.body_proportion.shoulder)),
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

        climber.left_arm = self.__find_hold_for_left_arm(
            climber, body_center)
        climber.right_arm = self.__find_hold_for_right_arm(
            climber, body_center)

    def __get_bottom_objects_fit_as_steps(self, max_distance_from_ground_in_cm: int) -> [DetectedObject]:
        max_distance_from_ground_in_px = (
            self.__marker.convert_cm_to_px(max_distance_from_ground_in_cm))

        # 40 cm of bottom boxes from image but exclude from left and right 15%
        return [obj for obj in self.__detected_objects if
                obj.bbox[3] > self.__img_height - max_distance_from_ground_in_px and
                obj.bbox[0] > 0.15 * self.__img_width and
                obj.bbox[2] < 0.85 * self.__img_width
                ]

    def __find_hold_for_left_arm(self, climber: Climber,
                                 body_center: int) -> BodyPart:
        # get holds available for left hand
        holds = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=climber.left_shoulder.start,
            radius=int(round(climber.body_proportion.arm))
        )

        # exclude holds that are on the right side of
        # the body_center
        holds = [hold for hold in holds
                 if hold.center.x < body_center]

        random_left_arm_hold_id = np.random.choice(len(holds), 1, replace=False)[0]
        random_left_arm_hold = holds[random_left_arm_hold_id]

        return BodyPart(
            start=climber.left_shoulder.start,
            end=random_left_arm_hold.center,
            color=Color.red(),
            thickness=10,
            detected_object=random_left_arm_hold
        )

    def __find_hold_for_right_arm(self, climber: Climber,
                                  body_center: int) -> BodyPart:
        # get holds available for right hand
        holds = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=climber.right_shoulder.end,
            radius=int(round(climber.body_proportion.arm))
        )

        # exclude holds that are on the left side of
        # the body_center
        holds = [hold for hold in holds
                 if hold.center.x > body_center]

        random_right_arm_hold_id = np.random.choice(len(holds), 1, replace=False)[0]
        random_right_arm_hold = holds[random_right_arm_hold_id]

        return BodyPart(
            start=climber.right_shoulder.end,
            end=random_right_arm_hold.center,
            color=Color.red(),
            thickness=10,
            detected_object=random_right_arm_hold
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

    def __is_climber_on_top(self, climber: Climber) -> bool:
        point_above_head = Point(
            x=climber.head.start.x,
            y=int(round(climber.head.start.y - climber.body_proportion.head * 2))
        )

        holds_around_point = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=point_above_head,
            radius=climber.body_proportion.arm
        )

        # exclude holds below point above head
        holds_around_point = list(filter(
            lambda hold: hold.center.y < point_above_head.y,
            holds_around_point
        ))

        return len(holds_around_point) == 0
