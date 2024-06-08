import pytest
import cv2
from src.aruco_marker import ArucoMarker
from src import config


def test_detect_marker() -> None:
    # given
    img = cv2.imread('./files/aruco_marker.jpeg')

    # when
    ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)

    # then
    assert True


def test_exception_when_no_marker_detected() -> None:
    # given
    img = cv2.imread('./files/no_aruco_marker.jpeg')

    # when and then
    with pytest.raises(ValueError):
        ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)


def test_get_width_in_cm() -> None:
    # given
    img = cv2.imread('./files/aruco_marker.jpeg')
    aruco_marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)

    # when
    width_in_cm = aruco_marker.get_width_in_cm()

    # then
    assert round(width_in_cm) == 7


def test_get_height_in_cm() -> None:
    # given
    img = cv2.imread('./files/aruco_marker.jpeg')
    aruco_marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)

    # when
    height_in_cm = aruco_marker.get_height_in_cm()

    # then
    assert round(height_in_cm) == 7


def test_get_pixels_per_centimeter() -> None:
    # given
    img = cv2.imread('./files/aruco_marker.jpeg')
    aruco_marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)

    # when
    pixels_per_centimeter = aruco_marker.get_pixels_per_centimeter()

    # then
    assert round(pixels_per_centimeter) == 79
