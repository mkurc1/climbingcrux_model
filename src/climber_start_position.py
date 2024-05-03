import numpy as np

from src.aruco_marker import ArucoMarker
from src.model.body_part import BodyPart
from src.model.body_proportion import BodyProportion
from src.model.climber import Climber
from src.model.color import Color
from src.model.detected_object import DetectedObject
from src import objects_detector, config


class ClimberStartPosition:
    def __init__(self, img_width: int, img_height: int, marker: ArucoMarker,
                 detected_objects: [DetectedObject]):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__marker = marker
        self.__detected_objects = detected_objects
        self.__climber = Climber()

    def prepare(self, climber_height_in_cm: int,
                starting_steps_max_distance_from_ground_in_cm: int) -> Climber:
        climber_height_in_px = self.__marker.convert_cm_to_pixel(climber_height_in_cm)
        body_proportion = BodyProportion(climber_height_in_px)

        bottom_objects = self.__get_bottom_objects_fit_as_steps(
            starting_steps_max_distance_from_ground_in_cm
        )

        starting_step_1_id = np.random.choice(len(bottom_objects), 1, replace=False)[0]
        starting_step_1 = bottom_objects[starting_step_1_id]

        # Get bottom holds in the circle but not the starting holds
        holds_in_circle = objects_detector.get_objects_around_point(
            detected_objects=bottom_objects,
            point=starting_step_1.get_center(),
            radius=self.__marker.convert_cm_to_pixel(config.STEP_RADIUS_IN_CM),
            exclude_detected_objects=[starting_step_1]
        )

        # Get one random hold in the circle as the second step
        starting_step_2_id = np.random.choice(len(holds_in_circle), 1, replace=False)[0]
        starting_step_2 = holds_in_circle[starting_step_2_id]

        # lower starting step
        lower_starting_step = starting_step_1 \
            if starting_step_1.get_center()[1] > starting_step_2.get_center()[1] \
            else starting_step_2

        position_between_starting_steps = (starting_step_1.get_center()[0] +
                                           starting_step_2.get_center()[0]) / 2

        top_head_point = (
            int(position_between_starting_steps),
            int(lower_starting_step.get_center()[1] - body_proportion.height))
        bottom_head_point = (
            int(position_between_starting_steps),
            int(lower_starting_step.get_center()[1] - body_proportion.height +
                body_proportion.head))

        self.__climber.head = BodyPart(
            start=top_head_point,
            end=bottom_head_point,
            color=Color(255, 255, 0),  # yellow color
            thickness=30
        )

        top_neck_point = (top_head_point[0], int(bottom_head_point[1]))
        bottom_neck_point = (
            bottom_head_point[0], int(bottom_head_point[1] + body_proportion.neck)
        )

        self.__climber.neck = BodyPart(
            start=top_neck_point,
            end=bottom_neck_point,
            color=Color(0, 125, 255),  # orange color
            thickness=10
        )

        top_trunk_point = (top_head_point[0], int(bottom_neck_point[1]))
        bottom_trunk_point = (bottom_head_point[0], int(bottom_neck_point[1] + body_proportion.trunk))

        self.__climber.trunk = BodyPart(
            start=top_trunk_point,
            end=bottom_trunk_point,
            color=Color.green(),  # green color
            thickness=30
        )

        start_left_shoulder_point = (
            bottom_neck_point[0] - int(body_proportion.shoulder),
            bottom_neck_point[1]
        )
        end_left_shoulder_point = (bottom_neck_point[0], bottom_neck_point[1])

        self.__climber.left_shoulder = BodyPart(
            start=start_left_shoulder_point,
            end=end_left_shoulder_point,
            color=Color.blue(),
            thickness=10
        )

        start_right_shoulder_point = (bottom_neck_point[0], bottom_neck_point[1])
        end_right_shoulder_point = (
            bottom_neck_point[0] + int(body_proportion.shoulder),
            bottom_neck_point[1]
        )

        self.__climber.right_shoulder = BodyPart(
            start=start_right_shoulder_point,
            end=end_right_shoulder_point,
            color=Color.blue(),
            thickness=10
        )

        if starting_step_1.get_center()[0] < starting_step_2.get_center()[0]:
            starting_left_step = starting_step_1
            starting_right_step = starting_step_2
        else:
            starting_left_step = starting_step_2
            starting_right_step = starting_step_1

        start_left_leg_point = (bottom_trunk_point[0], bottom_trunk_point[1])
        end_left_leg_point = starting_left_step.get_center()

        self.__climber.left_leg = BodyPart(
            start=start_left_leg_point,
            end=end_left_leg_point,
            color=Color.blue(),
            thickness=10
        )

        start_right_leg_point = (bottom_trunk_point[0], bottom_trunk_point[1])
        end_right_leg_point = starting_right_step.get_center()

        self.__climber.right_leg = BodyPart(
            start=start_right_leg_point,
            end=end_right_leg_point,
            color=Color.blue(),
            thickness=10
        )

        self.__climber.left_hand = self.__find_hold_for_left_hand(
            body_proportion, position_between_starting_steps)
        self.__climber.right_hand = self.__find_hold_for_right_hand(
            body_proportion, position_between_starting_steps)

        return self.__climber

    def __get_bottom_objects_fit_as_steps(self, max_distance_from_ground_in_cm: int) -> [DetectedObject]:
        max_distance_from_ground_in_px = (
            self.__marker.convert_cm_to_pixel(max_distance_from_ground_in_cm))

        # 40 cm of bottom boxes from image but exclude from left and right 15%
        return [obj for obj in self.__detected_objects if
                obj.bbox[3] > self.__img_height - max_distance_from_ground_in_px and
                obj.bbox[0] > 0.15 * self.__img_width and
                obj.bbox[2] < 0.85 * self.__img_width
                ]

    def __find_hold_for_left_hand(self, body_proportion: BodyProportion,
                                  position_between_starting_steps: tuple[int, int]) -> BodyPart:
        # get holds available for left hand
        holds = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=self.__climber.left_shoulder.start,
            radius=int(body_proportion.arm)
        )

        # exclude holds that are on the right side of
        # the position_between_starting_steps
        holds = [hold for hold in holds
                 if hold.get_center()[0] < position_between_starting_steps]

        random_left_hand_hold_id = np.random.choice(len(holds), 1, replace=False)[0]
        random_left_hand_hold = holds[random_left_hand_hold_id]

        return BodyPart(
            start=self.__climber.left_shoulder.start,
            end=random_left_hand_hold.get_center(),
            color=Color.red(),
            thickness=10
        )

    def __find_hold_for_right_hand(self, body_proportion: BodyProportion,
                                   position_between_starting_steps: tuple[int, int]) -> BodyPart:
        # get holds available for right hand
        holds = objects_detector.get_objects_around_point(
            detected_objects=self.__detected_objects,
            point=self.__climber.right_shoulder.end,
            radius=int(body_proportion.arm)
        )

        # exclude holds that are on the left side of
        # the position_between_starting_steps
        holds = [hold for hold in holds
                 if hold.get_center()[0] > position_between_starting_steps]

        random_right_hand_hold_id = np.random.choice(len(holds), 1, replace=False)[0]
        random_right_hand_hold = holds[random_right_hand_hold_id]

        return BodyPart(
            start=self.__climber.right_shoulder.end,
            end=random_right_hand_hold.get_center(),
            color=Color.red(),
            thickness=10
        )
