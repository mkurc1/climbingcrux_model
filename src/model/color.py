from dataclasses import dataclass


@dataclass(frozen=True)
class Color:
    __red: int
    __green: int
    __blue: int

    def __post_init__(self):
        if not (0 <= self.__red <= 255):
            raise ValueError("Red value must be between 0 and 255")
        if not (0 <= self.__green <= 255):
            raise ValueError("Green value must be between 0 and 255")
        if not (0 <= self.__blue <= 255):
            raise ValueError("Blue value must be between 0 and 255")

    def rgb(self) -> tuple[int, int, int]:
        return self.__red, self.__green, self.__blue

    def bgr(self) -> tuple[int, int, int]:
        return self.__blue, self.__green, self.__red

    @staticmethod
    def red() -> 'Color':
        return Color(255, 0, 0)

    @staticmethod
    def green() -> 'Color':
        return Color(0, 255, 0)

    @staticmethod
    def blue() -> 'Color':
        return Color(0, 0, 255)

    @staticmethod
    def black() -> 'Color':
        return Color(0, 0, 0)

    @staticmethod
    def white() -> 'Color':
        return Color(255, 255, 255)
