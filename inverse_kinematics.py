import numpy as np

try:
    from sim.joint_limits import clamp_dmp_vector
except ImportError:
    clamp_dmp_vector = None


class InverseKinematics:
    def __init__(self):
        self.running = False
        # Initiala offsets för kalibrering Elbow
        self.elbow_zero_offset = 0.0
        self.elbow_is_straight = True

        # Initiala offset för kalibrering Shoulder flexion
        self.upper_arm_neutral = None
        self.shoulder_flex_direction = None

        #initiela offset för kalibrering Shoulder abduction
        self.shoulder_abd_zero_offset = 0.0

        #combination of offsets for shoulder flexion/abduction calibration
        self.side_axis = np.array([1.0, 0.0, 0.0])
        self.forward_axis = np.array([0.0, 0.0, -1.0])

        #rotation
        self.shoulder_rotation_enabled = False

    def start(self):
        self.running = True
        print("Inverse kinematics started")

    def stop(self):
        self.running = False
        print("Inverse kinematics stopped")

    def _safe_point(self, p):
        if p is None:
            return None

        p = np.array(p, dtype=float)

        if np.any(np.isnan(p)):
            return None

        return p

    def pixel_to_camera_3d(self, p, fx=615, fy=615, cx=320, cy=240):
        if p is None:
            return None

        x, y, z = p

        if z is None:
            return None

        X = (x - cx) * z / fx
        Y = -(y - cy) * z / fy
        Z = z

        return np.array([X, Y, Z], dtype=float)

    def _angle_between(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return None

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.arccos(cos_angle))
    
    #-----------------------------
    # Kalibreringsmetoder
    #-----------------------------
    
    def calibrate_elbow_zero(self, elbow_angle):
        if elbow_angle is not None:
            self.elbow_zero_offset = elbow_angle
            print("Elbow zero calibrated:", np.degrees(elbow_angle), "deg")

    def calibrate_upper_arm_neutral(self, shoulder, elbow):
        shoulder = self.pixel_to_camera_3d(self._safe_point(shoulder))
        elbow = self.pixel_to_camera_3d(self._safe_point(elbow))

        if shoulder is None or elbow is None:
            return

        v = elbow - shoulder
        norm = np.linalg.norm(v)

        if norm == 0:
            return

        self.upper_arm_neutral = v / norm
        print("Upper arm neutral calibrated")

    def calibrate_shoulder_flex_direction(self, shoulder, elbow):
        shoulder = self.pixel_to_camera_3d(self._safe_point(shoulder))
        elbow = self.pixel_to_camera_3d(self._safe_point(elbow))

        if shoulder is None or elbow is None or self.upper_arm_neutral is None:
            return

        v = elbow - shoulder
        norm = np.linalg.norm(v)

        if norm == 0:
            return

        current_dir = v / norm
        direction = current_dir - self.upper_arm_neutral

        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            return

        self.shoulder_flex_direction = direction / direction_norm
        print("Shoulder flex direction calibrated")

    def calibrate_shoulder_abd_zero(self, shoulder_abd_angle):
        if shoulder_abd_angle is not None:
            self.shoulder_abd_zero_offset = shoulder_abd_angle
            print(
                "Shoulder abduction zero calibrated:",
                np.degrees(shoulder_abd_angle),
                "deg"
            )
    
    def calculate_shoulder_rotation_from_forearm(
        self,
        elbow,
        wrist,
        rotation_enabled,
        z_range=0.25,
    ):
        if not rotation_enabled:
            return 0.0

        elbow = self._safe_point(elbow)
        wrist = self._safe_point(wrist)

        if elbow is None or wrist is None:
            return 0.0

        forearm_z = wrist[2] - elbow[2]

        SHOULDER_ROT_MAX = np.radians(40)

        shoulder_rot = np.clip(
            forearm_z / z_range,
            -1.0,
            1.0
        ) * SHOULDER_ROT_MAX

        return shoulder_rot

        
    #-----------------------------
    # Huvudmetod för att beräkna armvinklar
    #----------------------------- 

    def calculate_arm_angles(self, shoulder, elbow, wrist):
        """
        Input:
            shoulder, elbow, wrist = [x, y, z]

        Output:
            q_rad = [
                elbow_flexion,
                shoulder_flexion,
                shoulder_abduction,
                shoulder_internal_rotation
            ]
        """

        if not self.running:
            return None

        shoulder = self._safe_point(shoulder)
        elbow = self._safe_point(elbow)
        wrist = self._safe_point(wrist)

        if shoulder is None or elbow is None or wrist is None:
            return None

        shoulder = self.pixel_to_camera_3d(shoulder)
        elbow = self.pixel_to_camera_3d(elbow)
        wrist = self.pixel_to_camera_3d(wrist)

        if shoulder is None or elbow is None or wrist is None:
            return None

        upper_arm = elbow - shoulder
        lower_arm = wrist - elbow

        # -----------------------------
        # 1. Elbow flexion
        # -----------------------------
        # Rak arm ≈ 0
        # Böjd arm ≈ större vinkel
        raw_elbow_angle = self._angle_between(upper_arm, lower_arm)

        if raw_elbow_angle is None:
            return None

        elbow_flexion = raw_elbow_angle

        elbow_flexion = elbow_flexion - self.elbow_zero_offset
        elbow_flexion = max(0.0, elbow_flexion)

        # Inför en "deadzone" för att undvika små rörelser när armen är nästan rak.
        
        ELBOW_ENTER_BEND = np.radians(12) #Now we do this insted of deadzone
        ELBOW_EXIT_BEND = np.radians(5)

        if self.elbow_is_straight:
            if elbow_flexion < ELBOW_ENTER_BEND:
                elbow_flexion = 0.0
            else:
                self.elbow_is_straight = False
        else:
            if elbow_flexion < ELBOW_EXIT_BEND:
                elbow_flexion = 0.0
                self.elbow_is_straight = True

        # Elbow max är 60 grader
        ROBOT_ELBOW_MAX = np.radians(60)

        elbow_flexion = np.clip(
            elbow_flexion,
            0.0,
            ROBOT_ELBOW_MAX
        )
        # Ytterligare en tröskel för att behandla armen som helhet
        ELBOW_STRAIGHT_THRESHOLD = np.radians(10)

        if elbow_flexion < ELBOW_STRAIGHT_THRESHOLD:
            elbow_flexion = 0.0
        
        # -----------------------------
        # 2. Shoulder flexion
        # -----------------------------
        # Baserat på hur mycket överarmen går fram/upp i bildplanet.
        # Arm rakt ner = 0 grader
        # Arm rakt fram = större vinkel
        # min 0, max 80 grader
        # OBS: första approximation.
        # -----------------------------
        # 3. Shoulder abduction
        # -----------------------------
        # Sidledsrörelse.
        # Arm min = 0 grader (rakt ner)
        # Arm max = 40 grade

        ux, uy, uz = upper_arm

        upper_arm_dir = upper_arm / (np.linalg.norm(upper_arm) + 1e-6)

        if self.upper_arm_neutral is not None:
            neutral = self.upper_arm_neutral
            side_axis = self.side_axis
            forward_axis = self.forward_axis

            neutral_component = np.dot(upper_arm_dir, neutral)
            side_component = np.dot(upper_arm_dir, side_axis)
            forward_component = np.dot(upper_arm_dir, forward_axis)

            shoulder_flexion = np.arctan2(
                max(0.0, forward_component),
                max(1e-6, neutral_component)
            )

            shoulder_abduction = np.arctan2(
                abs(side_component),
                max(1e-6, neutral_component)
            )

        else:
            shoulder_flexion = np.arctan2(-uy, abs(uz) + 1e-6)
            shoulder_abduction = np.arctan2(abs(ux), abs(uy) + 1e-6)


        SHOULDER_FLEX_DEADZONE = np.radians(5)
        if shoulder_flexion < SHOULDER_FLEX_DEADZONE:
            shoulder_flexion = 0.0

        SHOULDER_FLEX_MAX = np.radians(80)
        shoulder_flexion = np.clip(shoulder_flexion, 0.0, SHOULDER_FLEX_MAX)


        shoulder_abduction = shoulder_abduction - self.shoulder_abd_zero_offset
        shoulder_abduction = max(0.0, shoulder_abduction)

        SHOULDER_ABD_DEADZONE = np.radians(8)
        if shoulder_abduction < SHOULDER_ABD_DEADZONE:
            shoulder_abduction = 0.0

        SHOULDER_ABD_MAX = np.radians(40)
        shoulder_abduction = np.clip(shoulder_abduction, 0.0, SHOULDER_ABD_MAX)

        # -----------------------------
        # 4. Shoulder internal rotation
        # -----------------------------
        # Limiteras så den inte påverkar de andra lederna så mycket. Kan ev. förbättras i framtiden med fler sensorer eller ML-modell.
        # roterar endast när elbow är max flexad och överarmen är i en position rakt ner.
        # Sätts till 0 tills vidare.

        upper_arm_down = shoulder_flexion < np.radians(10) and shoulder_abduction < np.radians(10)
        elbow_fully_flexed = elbow_flexion > np.radians(50)

        rotation_enabled = upper_arm_down and elbow_fully_flexed

        shoulder_internal_rotation = self.calculate_shoulder_rotation_from_forearm(
            elbow,
            wrist,
            rotation_enabled
        )
        
        #-----------------------------
        # Samla alla vinklar i en array och konvertera till grader
        #-----------------------------

        q_rad = np.array([
            elbow_flexion,
            shoulder_flexion,
            shoulder_abduction,
            shoulder_internal_rotation
        ], dtype=float)

        # Gör negativa värden till 0 för flex/abd innan clamp
        q_rad[0] = max(0.0, q_rad[0])
        q_rad[1] = max(0.0, q_rad[1])
        q_rad[2] = max(0.0, q_rad[2])

        # Använd simulatorns joint limits om de finns
        if clamp_dmp_vector is not None:
            q_rad = clamp_dmp_vector(q_rad)

        q_deg = np.degrees(q_rad)

        return {
            "q_rad": q_rad,
            "q_deg": q_deg,
            "elbow_flexion_rad": q_rad[0],
            "shoulder_flexion_rad": q_rad[1],
            "shoulder_abduction_rad": q_rad[2],
            "shoulder_internal_rotation_rad": q_rad[3],
            "elbow_flexion_deg": q_deg[0],
            "shoulder_flexion_deg": q_deg[1],
            "shoulder_abduction_deg": q_deg[2],
            "shoulder_internal_rotation_deg": q_deg[3],
        }