import depthai as dai
import cv2
import mediapipe as mp
import numpy as np
import json
# -----------------------------
# MediaPipe setup You need to install mediapipe with: pip install mediapipe==0.10.14 if you have Python 3.12
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # motsvarar ungefär "Full"
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# DepthAI / OAK-D Lite setup
# -----------------------------
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera).build()
cam.setSensorType(dai.CameraSensorType.COLOR)

cam_out = cam.requestOutput(
    (640, 480),
    type=dai.ImgFrame.Type.BGR888p
)
# -----------------------------
# Stereo depth setup
# -----------------------------
mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

left_out = mono_left.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)
right_out = mono_right.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8)

stereo = pipeline.create(dai.node.StereoDepth)

left_out.link(stereo.left)
right_out.link(stereo.right)

depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
video_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()
print("OAK-D Lite is running. Press 'q' to quit.")

def get_depth_at_point(depth_frame, x, y, rgb_w, rgb_h):
    if x is None or y is None:
        return None

    depth_h, depth_w = depth_frame.shape

    # Skala RGB-koordinater till depth-bildens storlek
    dx = int(x * depth_w / rgb_w)
    dy = int(y * depth_h / rgb_h)

    # Liten ruta runt punkten istället för exakt en pixel
    radius = 6
    x1 = max(0, dx - radius)
    x2 = min(depth_w, dx + radius + 1)
    y1 = max(0, dy - radius)
    y2 = min(depth_h, dy + radius + 1)

    roi = depth_frame[y1:y2, x1:x2]

    valid_depths = roi[roi > 0]

    if len(valid_depths) == 0:
        return None

    return int(np.median(valid_depths))

recording = False
recorded_data = []
frame_count = 0

# -----------------------------
# Main loop
# -----------------------------
while pipeline.isRunning():
    frame_count += 1
    hand_wrist_pixel = None
    hand_keypoints = []
    frame_in = video_queue.get()
    frame = frame_in.getCvFrame()
    modolu = 10

    depth_in = depth_queue.tryGet()
    depth_frame = None

    if depth_in is not None:
        depth_frame = depth_in.getFrame()

    if frame is None:
        continue

    # MediaPipe vill ha RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape

    # Kör hand tracking
    results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    # Rita skelett
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            label = handedness.classification[0].label  # "Left" eller "Right"

            # FILTRERA - bara vänster hand
            if label != "Right":
                continue
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Om du även vill rita tydligare punkter själv:
            for i, lm in enumerate(hand_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
            
                depth_mm = None
                if depth_frame is not None:
                    depth_mm = get_depth_at_point(depth_frame, x, y, w, h)

                depth_m = depth_mm / 1000 if depth_mm is not None else None

                hand_keypoints.append({
                    "id": i,
                    "x": x,
                    "y": y,
                    "depth_m": depth_m
                })
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                
                # Spara handens handled
                if i == mp_hands.HandLandmark.WRIST:
                    hand_wrist_pixel = (x, y)

    # Rita pose skelett
    if pose_results.pose_landmarks:
        h, w, _ = frame.shape
        lm = pose_results.pose_landmarks.landmark

        # Vänster arm
        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]

        if shoulder.visibility > 0.5 and elbow.visibility > 0.5 and wrist.visibility > 0.5:
            sx, sy = int(shoulder.x * w), int(shoulder.y * h)
            ex, ey = int(elbow.x * w), int(elbow.y * h)
            # Använd handens handled om den finns, annars pose handled
            if hand_wrist_pixel is not None:
                wx, wy = hand_wrist_pixel
            elif wrist.visibility > 0.5:
                wx, wy = int(wrist.x * w), int(wrist.y * h)
            else:
                wx, wy = None, None

            # Rita ut koordinater i terminalen
            if depth_frame is not None:
                shoulder_depth = get_depth_at_point(depth_frame, sx, sy, w, h)
                elbow_depth = get_depth_at_point(depth_frame, ex, ey, w, h)
                wrist_depth = get_depth_at_point(depth_frame, wx, wy, w, h)
                
                if frame_count % modolu == 0:
                    print(json.dumps(hand_keypoints, indent=2))
                    if shoulder_depth is not None:
                        print(f"Shoulder: x={sx}, y={sy}, depth={shoulder_depth/1000:.3f} m")
                    else:
                        print(f"Shoulder: x={sx}, y={sy}, depth=None")

                    if elbow_depth is not None:
                        print(f"Elbow:    x={ex}, y={ey}, depth={elbow_depth/1000:.3f} m")
                    else:
                        print(f"Elbow:    x={ex}, y={ey}, depth=None")
                    if wx is not None and wy is not None:
                        if wrist_depth is not None:
                            print(f"Wrist: x={wx}, y={wy}, depth={wrist_depth/1000:.3f} m")
                        else:
                            print(f"Wrist: x={wx}, y={wy}, depth=None")
                    print("-----")
                    print("Frame count:", frame_count)
                    print("-----")

                    if recording:
                        frame_data = {
                            "shoulder": [
                                sx, sy,
                                shoulder_depth / 1000 if shoulder_depth is not None else None
                            ],
                            "elbow": [
                                ex, ey,
                                elbow_depth / 1000 if elbow_depth is not None else None
                            ],
                            "wrist": [
                                wx, wy,
                                wrist_depth / 1000 if wrist_depth is not None else None
                            ],
                            "hand": hand_keypoints
                        }

                        recorded_data.append(frame_data)
                        
            # Rita punkter
            cv2.circle(frame, (sx, sy), 6, (0, 0, 255), -1)
            cv2.circle(frame, (ex, ey), 6, (0, 0, 255), -1)
            cv2.circle(frame, (wx, wy), 6, (0, 0, 255), -1)

            # Rita linjer
            cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 3)
            cv2.line(frame, (ex, ey), (wx, wy), (255, 255, 255), 3)

    frame = cv2.flip(frame, 1)  # Spegelvänd för mer naturlig interaktion 
    cv2.imshow("OAK-D Lite Hand Skeleton", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('k'):
        recording = not recording
        print("Recording:", recording)

        if not recording:
            with open("recording.json", "w") as f:
                output = {
                    "user_id": 1,
                    "sequence": len(recorded_data),
                    "data": recorded_data
                }  
                json.dump(output, f, indent=2)
            print("Saved recording.json")

    if key == ord('q'):
        break

hands.close()
pose.close()
cv2.destroyAllWindows()