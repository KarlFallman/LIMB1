import numpy as np


class KalmanPointFilter:
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=1e-1): 
        #Ökad process_noise för att sänka latencyn
        """
        Kalman filter för en punkt (x, y, z)

        dt = tid mellan frames
        process_noise = hur mycket vi tror systemet rör sig
        measurement_noise = hur brusig mätningen är
        """

        self.dt = dt

        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix (vi mäter bara position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Covariance matrix
        self.P = np.eye(6)

        # Process noise
        self.Q = process_noise * np.eye(6)

        # Measurement noise
        self.R = measurement_noise * np.eye(3)

        self.initialized = False

    def update(self, x, y, z=None, measurement_valid=True):
        """
        Om measurement_valid=True:
            använd ny mätning.
        Om measurement_valid=False:
            håll senaste värde, ingen prediction.
        """

        # Initiera första gången
        if not self.initialized:
            if x is None or y is None:
                return None, None, None

            if z is None:
                z = 0.0

            self.x[:3] = np.array([[x], [y], [z]])
            self.initialized = True
            return x, y, z

        # Om mätningen är dålig:
        # gör INTE prediction, håll senaste värde
        if not measurement_valid or x is None or y is None:
            return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])

        # Om z saknas men x,y finns:
        # använd senaste z
        if z is None:
            z = float(self.x[2, 0])

        # --- Prediction ---
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # --- Update ---
        z_meas = np.array([[x], [y], [z]])

        y_residual = z_meas - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])