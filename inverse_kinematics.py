import numpy as np

try:
    from sim.joint_limits import clamp_dmp_vector
except ImportError:
    clamp_dmp_vector = None


class InverseKinematics:
    def __init__(self):
        self.running = False

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

    def _angle_between(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return None

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.arccos(cos_angle))

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

        elbow_flexion = np.pi - raw_elbow_angle

        # -----------------------------
        # 2. Shoulder flexion
        # -----------------------------
        # Baserat på hur mycket överarmen går fram/upp i bildplanet.
        # OBS: första approximation.
        ux, uy, uz = upper_arm

        shoulder_flexion = np.arctan2(-uy, abs(uz) + 1e-6)

        # -----------------------------
        # 3. Shoulder abduction
        # -----------------------------
        # Sidledsrörelse.
        shoulder_abduction = np.arctan2(abs(ux), abs(uy) + 1e-6)

        # -----------------------------
        # 4. Shoulder internal rotation
        # -----------------------------
        # Svår att uppskatta från bara shoulder-elbow-wrist.
        # Sätts till 0 tills vidare.
        shoulder_internal_rotation = 0.0

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