import numpy as np

"""
Limitations:

| Model                    | Status     |
| ------------------------ | ---------- |
| No polarization mismatch | acceptable |
| No cable loss            | acceptable |
| No impedance mismatch    | acceptable |
| No Fresnel zone modeling | acceptable |
| No ground reflection     | acceptable |
| No near-field effects    | acceptable |
| No mutual coupling       | acceptable |
"""


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
        tx_power_dbm=40.0,
        tx_gain_dbi=10.5,
        distance_m=1500.0,
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
        self.tx_phi = 90.0

        # Initial receiver direction (degrees)
        self.tx_theta = 90.0
        self.tx_phi = 90.0

        # Shadowing is slow-varying
        self.shadowing_db = np.random.normal(0, self.shadow_std)

    # ---------------------- Physics Models ----------------------

    def _free_space_path_loss_1m(self):
        f_mhz = self.freq_ghz * 1000
        d_km = 0.001  # 1 m
        return 32.44 + 20 * np.log10(f_mhz) + 20 * np.log10(d_km)

    def path_loss(self):
        """Log-distance path loss with shadowing."""
        pl_1m = self._free_space_path_loss_1m()
        return pl_1m + 10 * self.n * np.log10(self.distance_m / 1.0) + self.shadowing_db

    def antenna_gain(self, rx_theta, rx_phi):
        """Receiver antenna gain (Yagi) in dBi."""

        # Relative angles
        delta_theta = rx_theta - self.tx_theta
        delta_phi = rx_phi - self.tx_phi

        main = self.g_max * np.exp(
            -0.5 * (delta_theta / self.sigma_theta) ** 2
            - 0.5 * (delta_phi / self.sigma_phi) ** 2
        )

        # Side lobes (azimuth only)
        side1 = (
            self.side_lobe_factor
            * self.g_max
            * np.exp(-0.5 * ((delta_theta + 60) / (1.8 * self.sigma_theta)) ** 2)
        )
        side2 = (
            self.side_lobe_factor
            * self.g_max
            * np.exp(-0.5 * ((delta_phi - 60) / (1.8 * self.sigma_theta)) ** 2)
        )

        return main + side1 + side2


    def interference(self):
        """Interference model: mostly quiet, occasional bursts."""
        if np.random.rand() < 0.05:  # 5% chance
            return np.random.uniform(-10, 5)
        return 0.0

    # ---------------------- Environment Step ----------------------

    def set_orientation(self, rx_theta, rx_phi):
        self.rx_theta = np.clip(rx_theta, 0, 180)
        self.rx_phi = np.clip(rx_phi, 0, 180)

    def measure_rssi(self, samples=10):
        return self._compute_rssi(self.rx_theta, self.rx_phi, samples)

    def _compute_rssi(self, rx_theta, rx_phi, samples):
        """
        Measure RSSI at given pan (theta) and tilt (phi).
        Returns averaged RSSI (dBm).
        """

        rssi_samples = []

        for _ in range(samples):
            rx_gain = self.antenna_gain(rx_theta, rx_phi)
            pl = self.path_loss()
            fast_fade = np.random.normal(0, self.fast_fade_std)
            interf = self.interference()
            noise = np.random.normal(0, self.meas_noise)

            rssi = (
                self.tx_power_dbm
                + self.tx_gain_dbi
                + rx_gain
                - pl
                + fast_fade
                + interf
                + noise
            )

            rssi_samples.append(rssi)

        return float(np.mean(rssi_samples))

    def step(self, action): ...

    def reset(self, rx_theta=90.0, rx_phi=90.0):
        self.rx_theta = rx_theta
        self.rx_phi = rx_phi
        self.reset_shadowing()
        return np.array(
            [self.rx_theta, self.rx_phi, self.measure_rssi()], dtype=np.float32
        )

    # ---------------------- Drift Models ----------------------

    def drift_transmitter(self, std_deg=0.05):
        """Slow random drift of transmitter direction."""
        self.tx_theta += np.random.normal(0, std_deg)
        self.tx_phi += np.random.normal(0, std_deg)

    def reset_shadowing(self):
        """Reset large-scale shadowing (environment change)."""
        self.shadowing_db = np.random.normal(0, self.shadow_std)
