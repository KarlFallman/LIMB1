import numpy as np


def calculate_grip_from_hand(hand_keypoints):
    if len(hand_keypoints) < 21:
        return 0.0

    wrist = np.array([hand_keypoints[0]["x"], hand_keypoints[0]["y"]], dtype=float)
    middle_mcp = np.array([hand_keypoints[9]["x"], hand_keypoints[9]["y"]], dtype=float)

    hand_scale = np.linalg.norm(middle_mcp - wrist)

    if hand_scale < 1e-6:
        return 0.0

    fingertip_ids = [4, 8, 12, 16, 20]
    normalized_distances = []

    for i in fingertip_ids:
        tip = np.array([hand_keypoints[i]["x"], hand_keypoints[i]["y"]], dtype=float)
        dist = np.linalg.norm(tip - wrist) / hand_scale
        normalized_distances.append(dist)

    avg_dist = np.mean(normalized_distances)

    OPEN_DIST = 2.0
    CLOSED_DIST = 1.0

    grip = (OPEN_DIST - avg_dist) / (OPEN_DIST - CLOSED_DIST)

    return float(np.clip(grip, 0.0, 1.0))


def calculate_finger_grips(hand_keypoints):
    if len(hand_keypoints) < 21:
        return None

    wrist = np.array([hand_keypoints[0]["x"], hand_keypoints[0]["y"]], dtype=float)
    middle_mcp = np.array([hand_keypoints[9]["x"], hand_keypoints[9]["y"]], dtype=float)

    hand_scale = np.linalg.norm(middle_mcp - wrist)

    if hand_scale < 1e-6:
        return None

    fingers = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    OPEN_DIST = 2.0
    CLOSED_DIST = 1.0

    finger_grips = {}

    for name, tip_id in fingers.items():
        tip = np.array([hand_keypoints[tip_id]["x"], hand_keypoints[tip_id]["y"]], dtype=float)
        dist = np.linalg.norm(tip - wrist) / hand_scale

        grip = (OPEN_DIST - dist) / (OPEN_DIST - CLOSED_DIST)
        finger_grips[name] = float(np.clip(grip, 0.0, 1.0))

    return finger_grips