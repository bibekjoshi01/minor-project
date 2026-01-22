import numpy as np


class AntennaEnvironment:
    """
    Realistic 2.4 GHz WiFi antenna environment model
    ------------------------------------------------
    - Receiver: 10.5 dBi Yagi antenna (pan + tilt)
    - Transmitter: fixed with slow angular drift
    - Includes path loss, shadowing, multipath, interference, and measurement noise
    All values are in dB / dBm domain.
    """

    def __init__(
        self,
        freq_ghz=2.45,
        tx_power_dbm=20.0,
        tx_gain_dbi=2.0,
        distance_m=10.0,
        path_loss_exp=2.5,
        shadow_std_db=4.0,
        fast_fade_std_db=2.0,
        meas_noise_db=1.0,
        g_max_dbi=10.5,
        sigma_theta=17.0,
        sigma_phi=12.0,
        side_lobe_factor=0.2,
    ):

        self.freq_ghz = freq_ghz
        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_dbi = tx_gain_dbi
        self.distance_m = distance_m
        self.n = path_loss_exp
        self.shadow_std = shadow_std_db
        self.fast_fade_std = fast_fade_std_db
        self.meas_noise = meas_noise_db

        self.g_max = g_max_dbi
        self.sigma_theta = sigma_theta
        self.sigma_phi = sigma_phi
        self.side_lobe_factor = side_lobe_factor

        # Initial transmitter direction (degrees)
        self.tx_theta = 90.0
        self.tx_phi = 0.0

        # Shadowing is slow-varying
        self.shadowing_db = np.random.normal(0, self.shadow_std)

    # ---------------------- Physics Models ----------------------

    def _free_space_path_loss_1m(self):
        """Free-space path loss at 1 meter (dB)."""
        return 32.44 + 20 * np.log10(self.freq_ghz)

    def path_loss(self):
        """Log-distance path loss with shadowing."""
        pl_1m = self._free_space_path_loss_1m()
        return pl_1m + 10 * self.n * np.log10(self.distance_m) + self.shadowing_db

    def antenna_gain(self, theta, phi):
        """Receiver antenna gain (Yagi) in dBi."""
        # Main lobe
        main = self.g_max * np.exp(
            -0.5 * ((theta - self.tx_theta) / self.sigma_theta) ** 2
            - 0.5 * ((phi - self.tx_phi) / self.sigma_phi) ** 2
        )

        # Side lobes (azimuth only)
        side1 = (
            self.side_lobe_factor
            * self.g_max
            * np.exp(
                -0.5 * ((theta - (self.tx_theta + 60)) / (1.8 * self.sigma_theta)) ** 2
            )
        )
        side2 = (
            self.side_lobe_factor
            * self.g_max
            * np.exp(
                -0.5 * ((theta - (self.tx_theta - 60)) / (1.8 * self.sigma_theta)) ** 2
            )
        )

        return main + side1 + side2

    def interference(self):
        """Interference model: mostly quiet, occasional bursts."""
        if np.random.rand() < 0.05:  # 5% chance
            return np.random.uniform(-10, 5)
        return 0.0

    # ---------------------- Environment Step ----------------------

    def step(self, theta, phi, samples=10):
        """
        Measure RSSI at given pan (theta) and tilt (phi).
        Returns averaged RSSI (dBm).
        """
        rssi_samples = []

        for _ in range(samples):
            pl = self.path_loss()
            gain = self.antenna_gain(theta, phi)
            fast_fade = np.random.normal(0, self.fast_fade_std)
            noise = np.random.normal(0, self.meas_noise)
            interf = self.interference()

            rssi = (
                self.tx_power_dbm
                + self.tx_gain_dbi
                + gain
                - pl
                + fast_fade
                + interf
                + noise
            )
            rssi_samples.append(rssi)

        return float(np.mean(rssi_samples))

    # ---------------------- Drift Models ----------------------

    def drift_transmitter(self, std_deg=0.05):
        """Slow random drift of transmitter direction."""
        self.tx_theta += np.random.normal(0, std_deg)
        self.tx_phi += np.random.normal(0, std_deg)

    def reset_shadowing(self):
        """Reset large-scale shadowing (environment change)."""
        self.shadowing_db = np.random.normal(0, self.shadow_std)
