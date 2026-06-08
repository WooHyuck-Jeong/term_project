class RRTNode:

    def __init__(self, x, y, z):

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        self.parent = None

        self.cost = 0.0

    def position(self):

        return [
            self.x,
            self.y,
            self.z
        ]

    def __repr__(self):

        return (
            f"RRTNode("
            f"x={self.x:.2f}, "
            f"y={self.y:.2f}, "
            f"z={self.z:.2f}, "
            f"cost={self.cost:.2f})"
        )