import depthai as dai
import cv2
import mediapipe as mp
import numpy as np
import json
from kalman_filter import KalmanPointFilter
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

# Reasonable limits for depth measurements (0.2m - 3m)                
def valid_depth(depth_mm):
    return depth_mm is not None and 200 < depth_mm < 3000

recording = False
recorded_data = []
frame_count = 0

wrist_filter = KalmanPointFilter()
elbow_filter = KalmanPointFilter()
shoulder_filter = KalmanPointFilter()
hand_filters = [KalmanPointFilter() for _ in range(21)]

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

                depth_valid = valid_depth(depth_mm)
                depth_m = depth_mm / 1000 if depth_valid else None

                fx, fy, fz = hand_filters[i].update(
                    x,
                    y,
                    depth_m,
                    measurement_valid=True
                )

                hand_keypoints.append({
                    "id": i,
                    "x": fx,
                    "y": fy,
                    "depth_m": fz
                })
                cv2.circle(frame, (int(fx), int(fy)), 4, (0, 255, 0), -1)
                
                # Spara handens handled
                if i == mp_hands.HandLandmark.WRIST:
                    hand_wrist_pixel = (int(fx), int(fy))

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

            if wx is None or wy is None:
                continue
            fsx, fsy, fsz = sx, sy, None
            fex, fey, fez = ex, ey, None
            fwx, fwy, fwz = wx, wy, None

            # Rita ut koordinater i terminalen
            if depth_frame is not None:
                shoulder_depth = get_depth_at_point(depth_frame, sx, sy, w, h)
                elbow_depth = get_depth_at_point(depth_frame, ex, ey, w, h)
                wrist_depth = get_depth_at_point(depth_frame, wx, wy, w, h)

                shoulder_z = shoulder_depth / 1000 if valid_depth(shoulder_depth) else None
                elbow_z = elbow_depth / 1000 if valid_depth(elbow_depth) else None
                wrist_z = wrist_depth / 1000 if valid_depth(wrist_depth) else None

                fsx, fsy, fsz = shoulder_filter.update(sx, sy, shoulder_z,measurement_valid=shoulder_z is not None)
                fex, fey, fez = elbow_filter.update(ex, ey, elbow_z,measurement_valid=elbow_z is not None)

                if hand_wrist_pixel is not None:
                    fwx, fwy = wx, wy

                    # Hämta depth från hand landmark 0, alltså wrist
                    if len(hand_keypoints) > 0:
                        fwz = hand_keypoints[0]["depth_m"]
                    else:
                        fwz = wrist_z
                else:
                    fwx, fwy, fwz = wrist_filter.update(
                        wx, wy, wrist_z,
                        measurement_valid=wrist_z is not None
                    )   
                
                if frame_count % modolu == 0:
                    print(json.dumps(hand_keypoints, indent=2))
                    if shoulder_depth is not None:
                        print(f"Shoulder: x={fsx:.3f}, y={fsy:.3f}, depth={fsz:.3f} m")
                    else:
                        print(f"Shoulder: x={fsx:.3f}, y={fsy:.3f}, depth=None")

                    if elbow_depth is not None:
                        print(f"Elbow:    x={fex:.3f}, y={fey:.3f}, depth={fez:.3f} m")
                    else:
                        print(f"Elbow:    x={fex:.3f}, y={fey:.3f}, depth=None")
                    if hand_wrist_pixel is not None:
                        if wrist_depth is not None:
                            print(f"Wrist: x={fwx:.3f}, y={fwy:.3f}, depth={fwz:.3f} m")
                        else:
                            print(f"Wrist: x={fwx}, y={fwy}, depth=None")
                    print("-----")
                    print("Frame count:", frame_count)
                    print("-----")

                    if recording:
                        frame_data = {
                            "shoulder": [fsx, fsy, fsz],
                            "elbow": [fex, fey, fez],
                            "hand": hand_keypoints
                        }

                        recorded_data.append(frame_data)
                        
            if fsx is None or fsy is None or fex is None or fey is None or fwx is None or fwy is None:
                continue

            # Rita punkter
            cv2.circle(frame, (int(fsx), int(fsy)), 6, (0, 255, 0), -1)
            cv2.circle(frame, (int(fex), int(fey)), 6, (0, 255, 0), -1)
            cv2.circle(frame, (int(fwx), int(fwy)), 6, (0, 255, 0), -1)

            cv2.line(frame, (int(fsx), int(fsy)), (int(fex), int(fey)), (255, 255, 255), 3)
            cv2.line(frame, (int(fex), int(fey)), (int(fwx), int(fwy)), (255, 255, 255), 3)

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