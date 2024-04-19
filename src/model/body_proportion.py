class BodyProportion:
    def __init__(self, height: int):
        self.height = height
        self.head = self.height / 8
        self.neck = self.head / 3
        self.shoulder = self.head * 2
        self.trunk = self.head * 2.6
        self.arm = self.head * 3.5
        self.leg = self.head * 4

