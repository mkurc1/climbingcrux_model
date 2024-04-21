class BodyPart:
    def __init__(self, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int = 10):
        self.start = start
        self.end = end
        self.color = color
        self.thickness = thickness
