import numpy as np
from base import AntennaEnvironmentBase


class AntennaEnvironmentSim(AntennaEnvironmentBase):
    """
    2D simulated antenna environment.
    Generates RSSI values based on a 2D Gaussian-like peak with noise and side lobes.
    """

    def __init__(
        self, pan_opt=90, tilt_opt=90, rssi_max=-30, rssi_min=-90, noise_std=2.0
    ):
        """
        Args:
            pan_opt: optimal pan angle (deg)
            tilt_opt: optimal tilt angle (deg)
            rssi_max: maximum RSSI at peak
            rssi_min: minimum RSSI
            noise_std: measurement noise std deviation
        """
        self.pan_opt = pan_opt
        self.tilt_opt = tilt_opt
        self.rssi_max = rssi_max
        self.rssi_min = rssi_min
        self.noise_std = noise_std

        self.pan = None
        self.tilt = None

    def set_orientation(self, pan, tilt):
        self.pan = int(np.clip(pan, 0, 180))
        self.tilt = int(np.clip(tilt, 0, 180))

    def measure_rssi(self, samples=1):
        """
        Simulate RSSI measurement:
        - Gaussian peak around (pan_opt, tilt_opt)
        - Added small side-lobes and noise
        """
        rssi_vals = []
        for _ in range(samples):
            # Base RSSI decay with distance from peak
            pan_delta = self.pan - self.pan_opt
            tilt_delta = self.tilt - self.tilt_opt
            rssi = (
                self.rssi_max - 0.5 * (pan_delta**2 + tilt_delta**2) / 25
            )  # shape factor

            # Add small random side-lobes
            side_lobe = np.random.choice([0, -3, -5], p=[0.7, 0.2, 0.1])
            rssi += side_lobe

            # Add Gaussian noise
            rssi += np.random.normal(0, self.noise_std)

            # Clip to min RSSI
            rssi = max(rssi, self.rssi_min)
            rssi_vals.append(rssi)

        return float(np.mean(rssi_vals))
