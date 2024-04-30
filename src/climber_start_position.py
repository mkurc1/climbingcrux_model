import numpy as np

from src.aruco_marker import ArucoMarker
from src.model.body_part import BodyPart
from src.model.body_proportion import BodyProportion
from src.model.climber import Climber
from src.model.color import Color
from src.model.detected_object import DetectedObject
from src import objects_detector, config


class ClimberStartPosition:
    def __init__(self, img_width: int, img_height: int, marker: ArucoMarker, detected_objects: [DetectedObject]):
        self.img_width = img_width
        self.img_height = img_height
        self.marker = marker
        self.detected_objects = detected_objects

    def prepare(self, climber_height_in_cm: int,
                starting_steps_max_distance_from_ground_in_cm: int) -> Climber:
        climber = Climber()
        climber_height_in_px = self.marker.convert_cm_to_pixel(climber_height_in_cm)
        body_proportion = BodyProportion(climber_height_in_px)

        starting_steps_max_distance_from_ground_in_px = (
            self.marker.convert_cm_to_pixel(starting_steps_max_distance_from_ground_in_cm))

        # 40 cm of bottom boxes from image but exclude from left and right 15%
        bottom_objects = [obj for obj in self.detected_objects if
                          obj.bbox[3] > self.img_height - starting_steps_max_distance_from_ground_in_px and
                          obj.bbox[0] > 0.15 * self.img_width and
                          obj.bbox[2] < 0.85 * self.img_width
                          ]

        random_starting_step_id = np.random.choice(len(bottom_objects), 1, replace=False)[0]
        random_starting_step = bottom_objects[random_starting_step_id]

        # Get bottom holds in the circle but not the starting holds
        holds_in_circle = objects_detector.get_objects_around_point(
            detected_objects=bottom_objects,
            point=random_starting_step.get_center(),
            radius=self.marker.convert_cm_to_pixel(config.STEP_RADIUS_IN_CM),
            exclude_detected_objects=[random_starting_step]
        )

        # Get one random hold in the circle as the second step
        random_second_step_id = np.random.choice(len(holds_in_circle), 1, replace=False)[0]
        random_second_step = holds_in_circle[random_second_step_id]

        starting_step_1 = random_starting_step
        starting_step_2 = random_second_step

        # lower starting step
        lower_starting_step = starting_step_1 if starting_step_1.get_center()[1] > starting_step_2.get_center()[
            1] else starting_step_2

        position_between_starting_steps = (starting_step_1.get_center()[0] + starting_step_2.get_center()[0]) / 2

        top_head_point = (
        int(position_between_starting_steps), int(lower_starting_step.get_center()[1] - body_proportion.height))
        bottom_head_point = (int(position_between_starting_steps),
                             int(lower_starting_step.get_center()[1] - body_proportion.height + body_proportion.head))

        climber.head = BodyPart(
            start=top_head_point,
            end=bottom_head_point,
            color=Color(255, 255, 0),  # yellow color
            thickness=30
        )

        top_neck_point = (top_head_point[0], int(bottom_head_point[1]))
        bottom_neck_point = (bottom_head_point[0], int(bottom_head_point[1] + body_proportion.neck))

        climber.neck = BodyPart(
            start=top_neck_point,
            end=bottom_neck_point,
            color=Color(0, 125, 255),  # orange color
            thickness=10
        )

        top_trunk_point = (top_head_point[0], int(bottom_neck_point[1]))
        bottom_trunk_point = (bottom_head_point[0], int(bottom_neck_point[1] + body_proportion.trunk))

        climber.trunk = BodyPart(
            start=top_trunk_point,
            end=bottom_trunk_point,
            color=Color.green(),  # green color
            thickness=30
        )

        start_left_shoulder_point = (bottom_neck_point[0] - int(body_proportion.shoulder), bottom_neck_point[1])
        end_left_shoulder_point = (bottom_neck_point[0], bottom_neck_point[1])

        climber.left_shoulder = BodyPart(
            start=start_left_shoulder_point,
            end=end_left_shoulder_point,
            color=Color.blue(),
            thickness=10
        )

        start_right_shoulder_point = (bottom_neck_point[0], bottom_neck_point[1])
        end_right_shoulder_point = (bottom_neck_point[0] + int(body_proportion.shoulder), bottom_neck_point[1])

        climber.right_shoulder = BodyPart(
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

        climber.left_leg = BodyPart(
            start=start_left_leg_point,
            end=end_left_leg_point,
            color=Color.blue(),
            thickness=10
        )

        start_right_leg_point = (bottom_trunk_point[0], bottom_trunk_point[1])
        end_right_leg_point = starting_right_step.get_center()

        climber.right_leg = BodyPart(
            start=start_right_leg_point,
            end=end_right_leg_point,
            color=Color.blue(),
            thickness=10
        )

        # get holds available for left hand
        holds_available_for_left_hand = objects_detector.get_objects_around_point(
            detected_objects=self.detected_objects,
            point=climber.left_shoulder.start,
            radius=int(body_proportion.arm)
        )

        # exclude holds that are on the right side of the position_between_starting_steps
        holds_available_for_left_hand = [hold for hold in holds_available_for_left_hand if
                                         hold.get_center()[0] < position_between_starting_steps]

        # get holds available for right hand
        holds_available_for_right_hand = objects_detector.get_objects_around_point(
            detected_objects=self.detected_objects,
            point=climber.right_shoulder.end,
            radius=int(body_proportion.arm)
        )

        # exclude holds that are on the left side of the position_between_starting_steps
        holds_available_for_right_hand = [hold for hold in holds_available_for_right_hand if
                                          hold.get_center()[0] > position_between_starting_steps]

        random_left_hand_hold_id = np.random.choice(len(holds_available_for_left_hand), 1, replace=False)[0]
        random_left_hand_hold = holds_available_for_left_hand[random_left_hand_hold_id]

        climber.left_hand = BodyPart(
            start=random_left_hand_hold.get_center(),
            end=climber.left_shoulder.start,
            color=Color.red(),
            thickness=10
        )

        random_right_hand_hold_id = np.random.choice(len(holds_available_for_right_hand), 1, replace=False)[0]
        random_right_hand_hold = holds_available_for_right_hand[random_right_hand_hold_id]

        climber.right_hand = BodyPart(
            start=climber.right_shoulder.end,
            end=random_right_hand_hold.get_center(),
            color=Color.red(),
            thickness=10
        )

        return climber
