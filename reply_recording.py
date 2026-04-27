import json
import time

with open("recording.json", "r") as f:
    recorded_data = json.load(f)

for frame_id, frame_data in enumerate(recorded_data):
    print(f"Frame {frame_id}")

    print("Shoulder:", frame_data["shoulder"])
    print("Elbow:   ", frame_data["elbow"])
    print("Wrist:   ", frame_data["wrist"])

    print("Hand keypoints:")
    for kp in frame_data["hand"]:
        print(kp)

    print("-----")

    time.sleep(0.05)  # ungefär 20 FPS