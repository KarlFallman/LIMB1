import depthai as dai
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands

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
    frame_in = video_queue.get()
    frame = frame_in.getCvFrame()

    if frame is None:
        continue

    # MediaPipe vill ha RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Kör hand tracking
    results = hands.process(frame_rgb)

    # Rita skelett
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Om du även vill rita tydligare punkter själv:
            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    cv2.imshow("OAK-D Lite Hand Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hands.close()
cv2.destroyAllWindows()