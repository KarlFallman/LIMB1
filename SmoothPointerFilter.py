class SmoothPointFilter:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.last = None

    def update(self, x, y, z=None, measurement_valid=True):
        if x is None or y is None:
            return self.last if self.last is not None else (None, None, None)

        if z is None:
            z = 0.0

        current = (x, y, z)

        if self.last is None:
            self.last = current
            return current

        if not measurement_valid:
            return self.last

        fx = self.alpha * x + (1 - self.alpha) * self.last[0]
        fy = self.alpha * y + (1 - self.alpha) * self.last[1]
        fz = self.alpha * z + (1 - self.alpha) * self.last[2]

        self.last = (fx, fy, fz)
        return self.last