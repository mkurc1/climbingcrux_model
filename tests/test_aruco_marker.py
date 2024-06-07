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
