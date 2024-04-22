# This file contains the BodyProportion class which is used to calculate the body proportions of a person based on their height.
# This class based on data from the website: https://hpc.anatomy4sculptors.com/

class BodyProportion:
    def __init__(self, height: int):
        self.height = height
        self.head = self.height / 8
        self.neck = self.head / 3
        self.shoulder = self.head
        self.trunk = self.head * 2.6
        self.arm = self.head * 3.5
        self.leg = self.head * 4
