import depthai as dai
import cv2
import mediapipe as mp
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

video_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)

pipeline.start()
print("OAK-D Lite is running. Press 'q' to quit.")

# -----------------------------
# Main loop
# -----------------------------
while pipeline.isRunning():
    hand_wrist_pixel = None
    frame_in = video_queue.get()
    frame = frame_in.getCvFrame()

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
                z = lm.z
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                print(f"Hand landmark {i}: x={x}, y={y}, z={z:.4f}")

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
            print(f"Shoulder: ({sx}, {sy})")
            print(f"Elbow:    ({ex}, {ey})")
            if wx is not None and wy is not None:
                print(f"Wrist:    ({wx}, {wy})")
            print("-----")

            # Rita punkter
            cv2.circle(frame, (sx, sy), 6, (0, 0, 255), -1)
            cv2.circle(frame, (ex, ey), 6, (0, 0, 255), -1)
            cv2.circle(frame, (wx, wy), 6, (0, 0, 255), -1)

            # Rita linjer
            cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 3)
            cv2.line(frame, (ex, ey), (wx, wy), (255, 255, 255), 3)

    frame = cv2.flip(frame, 1)  # Spegelvänd för mer naturlig interaktion 
    cv2.imshow("OAK-D Lite Hand Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hands.close()
pose.close()
cv2.destroyAllWindows()