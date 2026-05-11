import numpy as np

class InverseKinematics:
    def __init__(self):
        self.running = False

        # Exempel-limiteringar, ändra efter er robot
        self.elbow_min = 0
        self.elbow_max = 60

        self.shoulder_min = 0
        self.shoulder_max = 80

    def start(self):
        self.running = True
        print("Inverse kinematics started")

    def stop(self):
        self.running = False
        print("Inverse kinematics stopped")

    def clamp(self, value, min_value, max_value):
        if value is None:
            return None
        return max(min_value, min(value, max_value))


def calculate_angle(p1, p2, p3):
    """
    Räknar vinkeln vid p2.
    Exempel:
    p1 = shoulder
    p2 = elbow
    p3 = wrist
    """

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    v1 = p1 - p2
    v2 = p3 - p2

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return None

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_arm_angles(shoulder, elbow, wrist):
    """
    Returnerar grundläggande armvinklar.
    Punkterna ska vara [x, y, z].
    """

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    return {
        "elbow_angle": elbow_angle
    }