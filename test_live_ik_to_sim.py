import depthai as dai
import cv2
import mediapipe as mp
import numpy as np
import math
from kalman_filter import KalmanPointFilter
from inverse_kinematics import InverseKinematics
from sim.joint_limits import clamp_dmp_vector
import pybullet as p
import time
from hand_kinematics import calculate_finger_grips, calculate_hand_rotation_2d

# -----------------------------
# MediaPipe setup You need to install mediapipe with: pip install mediapipe==0.10.14 if you have Python 3.12
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Joint indices in PyBullet URDF for fingers
thumb_joints  = [9, 10, 11]
index_joints  = [12, 13, 14]
middle_joints = [15, 16, 17]
ring_joints   = [19, 20, 21]
pinky_joints  = [23, 24, 25]

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, # 0 är lättare modell, 1 är full, 2 är heavy. För live tracking kan det vara värt att testa 0 för att minska latensen  
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # motsvarar ungefär "Full"
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
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(640, 480)

left_out.link(stereo.left)
right_out.link(stereo.right)

depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False) # Var 4 men testar byta till 1 för att se om det minskar latensen, risk för att missa frames dock
video_queue = cam_out.createOutputQueue(maxSize=1, blocking=False)      

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

# -----------------------------
# PyBullet setup
# -----------------------------

def joint_index(body_uid, joint_name):
    for i in range(p.getNumJoints(body_uid)):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8")
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found: {joint_name}")


def set_pose(robot, sh_rotz, sh_roty, sh_rotx, elbow_roty, wrist_rotx, wrist_rotz,
             sh_rot=0.0, sh_flex=0.0, sh_abd=0.0, elbow=0.0,  wrist_x=0.0, wrist_z=0.0):

    ROBOT_ELBOW_MAX = 1.05  # ca 60 grader i radianer
    ELBOW_GAIN = 1.5
    SHOULDER_FLEX_SIM_ZERO = -1.39  # testa först att kalibrera detta istället för att hårdkoda

    elbow = np.clip(elbow * ELBOW_GAIN, 0.0, ROBOT_ELBOW_MAX)
   
    p.resetJointState(robot, sh_rotz, sh_rot) # shoulder rotation
    p.resetJointState(robot, sh_roty, SHOULDER_FLEX_SIM_ZERO + sh_flex) # shoulder flexion
    p.resetJointState(robot, sh_rotx, sh_abd) # shoulder abduktion
    p.resetJointState(robot, elbow_roty, elbow) # elbow flexion

    p.resetJointState(robot, wrist_rotx, wrist_x) # wrist rotation
    p.resetJointState(robot, wrist_rotz, wrist_z) # wrist rotation

p.connect(p.GUI)
p.setGravity(0, 0, 0)

robot = p.loadURDF(
    "sim/arm/left_arm.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
)

sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
sh_roty = joint_index(robot, "jLeftShoulder_roty")
elbow_roty = joint_index(robot, "jLeftElbow_roty")
wrist_rotx = joint_index(robot, "jLeftWrist_rotx")
wrist_rotz = joint_index(robot, "jLeftWrist_rotz")

num_joints = p.getNumJoints(robot)
for i in range(-1, num_joints):
    for j in range(-1, num_joints):
        p.setCollisionFilterPair(robot, robot, i, j, enableCollision=0)

wrist_filter = KalmanPointFilter()
elbow_filter = KalmanPointFilter()
shoulder_filter = KalmanPointFilter()
hand_filters = [KalmanPointFilter() for _ in range(21)]
ik = InverseKinematics()
ik.start()

#-----------------------------
# Inverse kinematics main function fingers
#-----------------------------

def set_hand_grips(robot, finger_grips):
    if finger_grips is None:
        return

    thumb = finger_grips["thumb"]
    index = finger_grips["index"]
    middle = finger_grips["middle"]
    ring = finger_grips["ring"]
    pinky = finger_grips["pinky"]

    thumb_angles = [
        -thumb * 0.5,
        -thumb * 0.6,
        -thumb * 0.8,
    ]

    index_angle = -index * 0.8
    middle_angle = -middle * 0.8
    ring_angle = -ring * 0.8
    pinky_angle = -pinky * 0.8

    for joint, angle in zip(thumb_joints, thumb_angles):
        p.resetJointState(robot, joint, angle)

    for j in index_joints:
        p.resetJointState(robot, j, index_angle)

    for j in middle_joints:
        p.resetJointState(robot, j, middle_angle)

    for j in ring_joints:
        p.resetJointState(robot, j, ring_angle)

    for j in pinky_joints:
        p.resetJointState(robot, j, pinky_angle)

#-----------------------------
# DMP-smoothing class
#-----------------------------

class LiveDMP:
    """
    A real-time continuous attractor based on DMP dynamics.
    Acts as a perfectly critically damped spring to smooth live, jittery targets.
    """
    def __init__(self, num_dofs=4, alpha_z=25.0):
        # Removed dt from init because we calculate it dynamically per frame nu
        self.num_dofs = num_dofs
        self.alpha_z = alpha_z
        self.beta_z = self.alpha_z / 4.0  
        
        self.y = None      
        self.dy = np.zeros(num_dofs) 
        
    def step(self, target_g, dt):
        # Safety net: If PC stutters and dt is massive, cap it so physics don't explode
        dt = np.clip(dt, 0.001, 0.1)
        
        target_g = np.array(target_g, dtype=float)
        
        if self.y is None:
            self.y = target_g.copy()
            return self.y
            
        steps = int(math.ceil(dt / 0.005))
        sub_dt = dt / steps

        for _ in range(steps):
            ddy = self.alpha_z * (self.beta_z * (target_g - self.y) - self.dy)
            self.dy += ddy * sub_dt
            self.y += self.dy * sub_dt

        return self.y.copy()

q_human = None

# Initialize TWO DMPs: One for the 4 arm joints, one for the 5 fingers
live_arm_dmp = LiveDMP(num_dofs=4, alpha_z=25.0)
live_finger_dmp = LiveDMP(num_dofs=5, alpha_z=30.0) # Fingers can be slightly snappier

# Start the clock for dynamic dt tracking
prev_time = time.time()

# -----------------------------
# Main loop
# -----------------------------
while pipeline.isRunning():
    # Calculate exact dynamic time (dt) for this frame
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    hand_wrist_pixel = None
    hand_keypoints = []
    frame_in = video_queue.get() #från get till tryget för att undvika att blocka om inget
    if frame_in is None:
        continue 

    frame = frame_in.getCvFrame()

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

        if shoulder.visibility > 0.5 and elbow.visibility > 0.5 and (hand_wrist_pixel is not None or wrist.visibility > 0.5):
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
                shoulder_point = [fsx, fsy, fsz]
                elbow_point = [fex, fey, fez]
                wrist_point = [fwx, fwy, fwz]

                angles = ik.calculate_arm_angles(
                    shoulder_point,
                    elbow_point,
                    wrist_point
                )
                if angles is not None:
                    q_human = angles["q_rad"]
                    q_robot = q_human.copy()

                    q_robot = clamp_dmp_vector(q_robot)

                     # 1. Pass BOTH the target and the dt
                    q_smooth = live_arm_dmp.step(q_robot, dt)
                    # 2. Clamp it AGAIN to prevent overshoot crashing the robot
                    q_robot = clamp_dmp_vector(q_smooth)

                    prev_hand_rot = None
                    ROT_ALPHA = 0.2
                    hand_rot = calculate_hand_rotation_2d(hand_keypoints)
                    if prev_hand_rot is None:
                        smooth_rot = hand_rot
                    else:
                        smooth_rot = ROT_ALPHA * hand_rot + (1 - ROT_ALPHA) * prev_hand_rot

                    prev_hand_rot = smooth_rot
                    hand_rot = smooth_rot

                    forearm = np.array(wrist_point) - np.array(elbow_point)
 
                    set_pose(
                        robot,
                        sh_rotz,
                        sh_roty,
                        sh_rotx,
                        elbow_roty,
                        wrist_rotx, 
                        wrist_rotz,
                        elbow=float(q_robot[0]),
                        sh_flex=float(q_robot[1]),
                        sh_abd=float(q_robot[2]),
                        sh_rot=0.0,#float(q_robot[3]),
                        wrist_x=float(-hand_rot), 
                        wrist_z=0.0       
                    )
                    
                    # ----- FINGER SMOOTHING -----
                    if len(hand_keypoints) >= 21:
                        raw_grips = calculate_finger_grips(hand_keypoints)
                        
                        # Convert dict to array for the DMP
                        target_fingers = np.array([
                            raw_grips["thumb"], 
                            raw_grips["index"], 
                            raw_grips["middle"], 
                            raw_grips["ring"], 
                            raw_grips["pinky"]
                        ])
                        
                        # Smooth it!
                        smooth_fingers = live_finger_dmp.step(target_fingers, dt)
                        
                        # Clamp grip percentages between 0.0 (open) and 1.0 (closed)
                        smooth_fingers = np.clip(smooth_fingers, 0.0, 1.0)
                        
                        # Repackage into dict
                        finger_grips = {
                            "thumb": smooth_fingers[0],
                            "index": smooth_fingers[1],
                            "middle": smooth_fingers[2],
                            "ring": smooth_fingers[3],
                            "pinky": smooth_fingers[4]
                        }

                        set_hand_grips(robot, finger_grips)
                    
                    p.stepSimulation()


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

    if key == ord('c'):
        if q_human is not None:

            # Kalibrera elbow-zero separat
            ik.calibrate_elbow_zero(
                angles["elbow_flexion_rad"]
            )

            # Kalibrera shoulder flex-zero separat
            ik.calibrate_upper_arm_neutral(shoulder_point, elbow_point)

            # Kalibrera shoulder abduction zero separat
            ik.calibrate_shoulder_abd_zero(
                angles["shoulder_abduction_rad"]
            )

            print("Joint offsets calibrated based on current pose.")

        else:
            print("No IK angle available for calibration")

    if key == ord('q'):
        ik.stop()
        p.disconnect()
        break

hands.close()
pose.close()
cv2.destroyAllWindows()