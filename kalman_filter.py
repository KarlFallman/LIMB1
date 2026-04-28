import numpy as np


class KalmanPointFilter:
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
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

    def update(self, x, y, z=None):
        """
        Uppdatera filtret med ny mätning
        """

        if z is None:
            z = 0.0

        z_meas = np.array([[x], [y], [z]])

        # Första gången → initiera
        if not self.initialized:
            self.x[:3] = z_meas
            self.initialized = True
            return x, y, z

        # --- Prediction ---
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # --- Update ---
        y_residual = z_meas - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return float(self.x[0]), float(self.x[1]), float(self.x[2])