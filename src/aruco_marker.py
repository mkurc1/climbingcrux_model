import math
import cv2
import imutils

from src.model.color import Color


class ArucoMarker:
    def __init__(self, aruco_dict: int, image: cv2.typing.MatLike, marker_perimeter_in_cm: float):
        img_tmp = imutils.resize(image.copy(), width=1216)
        gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        params = cv2.aruco.DetectorParameters()
        markers = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[0]

        if len(markers) == 0:
            raise ValueError("No ArUco marker detected")

        self.corners = markers[0]
        self.marker_perimeter_in_cm = marker_perimeter_in_cm

        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        top_left, top_right, bottom_right, bottom_left = self.corners.reshape((4, 2))

        # convert each of the (x, y)-coordinate pairs to integers
        self.top_left = (int(top_left[0]), int(top_left[1]))
        self.top_right = (int(top_right[0]), int(top_right[1]))
        self.bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        self.bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

    def get_perimeter(self) -> float:
        return cv2.arcLength(self.corners, True)

    def get_pixels_per_centimeter(self) -> float:
        return self.get_perimeter() / self.marker_perimeter_in_cm

    def get_pixel_per_meter(self) -> float:
        return self.get_pixels_per_centimeter() * 100

    def get_width(self) -> float:
        return math.dist(self.top_left, self.top_right)

    def get_height(self) -> float:
        return math.dist(self.top_right, self.bottom_right)

    def get_center(self) -> tuple[int, int]:
        return (
            int((self.top_left[0] + self.bottom_right[0]) / 2),
            int((self.top_left[1] + self.bottom_right[1]) / 2)
        )

    def get_width_in_cm(self) -> float:
        return self.get_width() / self.get_pixels_per_centimeter()

    def get_height_in_cm(self) -> float:
        return self.get_height() / self.get_pixels_per_centimeter()

    def convert_cm_to_px(self, cm: float) -> int:
        return int(cm * self.get_pixels_per_centimeter())

    def convert_px_to_cm(self, px: int) -> float:
        return px / self.get_pixels_per_centimeter()

    def draw_bounding_box(self, image: cv2.typing.MatLike, color: Color,
                          thickness: int = 2) -> None:
        cv2.line(image, self.top_left, self.top_right, color.bgr(), thickness)
        cv2.line(image, self.top_right, self.bottom_right, color.bgr(), thickness)
        cv2.line(image, self.bottom_right, self.bottom_left, color.bgr(), thickness)
        cv2.line(image, self.bottom_left, self.top_left, color.bgr(), thickness)
