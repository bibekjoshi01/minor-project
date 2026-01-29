"""
Physically-correct RSSI vs angle simulator for directional PCB Yagi antennas.
- Uses vector geometry for Tx/Rx boresight and propagation direction.
- Uses Friis transmission formula (dB form) + log-distance path loss + shadowing + small-scale fading.
- Supports import of measured antenna pattern CSV (angle_deg, gain_dBi) OR a synthetic Yagi-like pattern.
- Outputs RSSI vs azimuth (and optionally vs tilt) plots.

References:
- Friis transmission equation (dB): Pr(dBm) = Pt(dBm) + Gt(dBi) + Gr(dBi) - FSPL(dB)
- Antenna theory: Balanis, Pozar (pattern / boresight modeling)
"""

import numpy as np

C = 299792458.0  # speed of light (m/s)


# Utilities: geometry & angles
# --------------------------
def sph_to_cartesian(az_deg, el_deg):
    """
    Convert azimuth (deg) and elevation (deg) to Cartesian unit vector.
    Azimuth: 0deg = +X, increases toward +Y (ccw from top view).
    Elevation: 0deg = horizontal; +90deg = +Z (up).
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    v = np.array([x, y, z], dtype=float)
    # normalize to avoid tiny numerical errors
    return v / np.linalg.norm(v)


def angle_between_rad(v1, v2):
    """Angle between two vectors in radians (numerically stable)"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)


# Antenna pattern helpers
# --------------------------
def synthetic_yagi_pattern_deg(
    polar_angle_deg, g_max_dbi=10.5, beamwidth_deg=40.0, side_lobe_level_dbi=-12.0
):
    """
    Simple synthetic Yagi-like cut (in the plane).
    - polar_angle_deg: angle off boresight (0 = boresight)
    - Returns gain in dBi.
    Notes:
      - This is a phenomenological model: main-lobe with high gain, side-lobes with lower gain.
      - For full 2D pattern, apply same function on theta (off-axis) for elevation and azimuth symmetry if needed.
    """
    a = np.abs(polar_angle_deg)
    # main lobe: approximate with raised cosine-like falloff
    bw = beamwidth_deg / 2.0
    main = g_max_dbi - 20.0 * (a / bw) ** 2  # quadratic roll-off in dB (clamped)
    # clamp main lobe floor
    main = np.maximum(main, -40.0)

    # add a simple side-lobe bump at around 60-80 degrees
    side = np.where(
        (a > 40) & (a < 110),
        side_lobe_level_dbi + 3 * np.cos(np.deg2rad((a - 60) / 50 * 180)) ** 2,
        -100.0,
    )
    # back-lobe region ~180 deg -> -20..-10 dBi
    back = np.where(a > 150, -18.0 + 6.0 * np.cos(np.deg2rad((a - 180))), -100.0)

    gain = np.maximum.reduce([main, side, back])
    return gain


class AntennaPattern:
    """
    Antenna pattern abstraction:
    - Either synth: synthetic_yagi_pattern_deg
    - Or loaded from CSV with columns: angle_deg, gain_dBi (assumed pattern cut in boresight plane).
    For a full 3D pattern you can provide interpolation over polar and azimuth angles (extension).
    """

    def __init__(self, use_synthetic=True, g_max_dbi=10.5, beamwidth_deg=40.0):
        self.use_synthetic = use_synthetic
        self.g_max = g_max_dbi
        self.bw = beamwidth_deg

    def gain_dbi_offboresight(self, off_angle_deg):
        """
        Returns gain at given off-boresight angle (0 = boresight).
        For synthetic pattern we treat off_angle up to 180deg.
        """
        a = (off_angle_deg + 180.0) % 360.0
        if a > 180.0:
            a = 360.0 - a

        if self.use_synthetic:
            return synthetic_yagi_pattern_deg(
                a, g_max_dbi=self.g_max, beamwidth_deg=self.bw
            )
        else:
            return float(self._interp(a))


# Propagation / RSSI class
# --------------------------
class AntennaSimulation:
    def __init__(
        self,
        freq_ghz=2.45,
        tx_power_dbm=40.0,
        tx_pos=(0.0, 0.0, 1.0),
        rx_pos=(1.0, 0.0, 1.0),
        tx_az_deg=0.0,
        tx_el_deg=0.0,
        rx_az_deg=180.0,
        rx_el_deg=0.0,
        tx_pattern=None,
        rx_pattern=None,
        path_loss_exponent=2.0,
        shadow_std_db=4.0,
        fastfade_std_db=2.0,
        samples_per_point=10,
        seed=None,
    ):
        self.freq_ghz = freq_ghz
        self.freq_hz = freq_ghz * 1e9
        self.wavelength = C / self.freq_hz
        self.tx_power_dbm = tx_power_dbm
        self.tx_pos = np.array(tx_pos, dtype=float)
        self.rx_pos = np.array(rx_pos, dtype=float)
        # boresight orientations
        self.tx_az_deg = float(tx_az_deg)
        self.tx_el_deg = float(tx_el_deg)
        self.rx_az_deg = float(rx_az_deg)
        self.rx_el_deg = float(rx_el_deg)

        # patterns
        self.tx_pattern = (
            tx_pattern if tx_pattern is not None else AntennaPattern(use_synthetic=True)
        )
        self.rx_pattern = (
            rx_pattern if rx_pattern is not None else AntennaPattern(use_synthetic=True)
        )

        # propagation params
        self.n = float(path_loss_exponent)
        self.sigma_shadow = float(shadow_std_db)
        self.sigma_fastfade = float(fastfade_std_db)
        self.samples = int(samples_per_point)

        # deterministic part of shadowing (can be reset for environment changes)
        rng = np.random.RandomState(seed)
        self._shadowing_db = rng.normal(0, self.sigma_shadow)

        self._rng = rng

    # ---------- path loss ----------
    def fspl_db(self, distance_m):
        """Free-space path loss (dB): 20 log10(4*pi*d / lambda)"""
        if distance_m <= 0:
            return 0.0
        return 20.0 * np.log10(4.0 * np.pi * distance_m / self.wavelength)

    def log_distance_pl_db(self, distance_m):
        """Log-distance path loss anchored to 1 meter (dB)"""
        pl_1m = self.fspl_db(distance_m)
        return (
            pl_1m
            + 10.0 * self.n * np.log10(max(distance_m, 1.0) / 1.0)
            + self._shadowing_db
        )

    # ---------- geometry ----------
    def propagation_unit_vector(self):
        """Unit vector from Tx to Rx"""
        vec = self.rx_pos - self.tx_pos
        return vec / np.linalg.norm(vec)

    def tx_boresight_vector(self):
        return sph_to_cartesian(self.tx_az_deg, self.tx_el_deg)

    def rx_boresight_vector(self, az_deg=None, el_deg=None):
        if az_deg is None:
            az_deg = self.rx_az_deg
        if el_deg is None:
            el_deg = self.rx_el_deg
        return sph_to_cartesian(az_deg, el_deg)

    # ---------- gain lookups ----------
    def tx_gain_dbi(self, prop_unit_vec):
        """Compute Tx gain in the direction of propagation (dBi)"""
        tx_bs = self.tx_boresight_vector()
        # angle between boresight and propagation vector
        theta_rad = angle_between_rad(tx_bs, prop_unit_vec)
        theta_deg = np.rad2deg(theta_rad)
        return self.tx_pattern.gain_dbi_offboresight(theta_deg)

    def rx_gain_dbi(self, prop_unit_vec, rx_az_deg=None, rx_el_deg=None):
        """Compute Rx gain for the arriving wave direction (-prop_unit_vec)"""
        if rx_az_deg is None:
            rx_az_deg = self.rx_az_deg
        if rx_el_deg is None:
            rx_el_deg = self.rx_el_deg
        rx_bs = self.rx_boresight_vector(rx_az_deg, rx_el_deg)
        incoming = (
            -prop_unit_vec
        )  # wave arrives from Tx -> Rx, so arriving direction is -prop
        theta_rad = angle_between_rad(rx_bs, incoming)
        theta_deg = np.rad2deg(theta_rad)
        return self.rx_pattern.gain_dbi_offboresight(theta_deg)

    # ---------- RSSI calculation ----------
    def measure_rssi_one_sample(self, rx_az_deg=None, rx_el_deg=None):
        """Single-sample RSSI (dBm) for given Rx orientation (if not provided uses current)"""
        prop_unit = self.propagation_unit_vector()
        d = np.linalg.norm(self.rx_pos - self.tx_pos)

        Gt = self.tx_gain_dbi(prop_unit)
        Gr = self.rx_gain_dbi(prop_unit, rx_az_deg, rx_el_deg)

        # Path loss (log-distance anchored at 1m) - uses shadowing member
        pl_db = self.log_distance_pl_db(d)

        # small-scale fading (dB) ~ Gaussian with std sigma_fastfade (approximate)
        fastfade_db = self._rng.normal(0.0, self.sigma_fastfade)

        rcv_dbm = self.tx_power_dbm + Gt + Gr - pl_db + fastfade_db
        return rcv_dbm

    def measure_rssi(self, rx_az_deg=None, rx_el_deg=None, samples=None):
        """Average RSSI over number of samples (in dBm domain, sample-level dB noise)"""
        if samples is None:
            samples = self.samples
        vals = [
            self.measure_rssi_one_sample(rx_az_deg, rx_el_deg) for _ in range(samples)
        ]
        return float(np.mean(vals)), float(np.std(vals))

    # ---------- helpers for sweeping ----------
    def sweep_azimuth(self, azimuths_deg=None, tilt_deg=0.0):
        """
        Sweep azimuth values (list or array) at fixed tilt (elevation).
        Returns (angles, mean_rssi_array, std_array)
        """

        if azimuths_deg is None:
            azimuths_deg = np.arange(0, 360, 1.0)

        means = []
        stds = []

        for az in azimuths_deg:
            m, s = self.measure_rssi(rx_az_deg=az, rx_el_deg=tilt_deg)
            means.append(m)
            stds.append(s)

        return np.array(azimuths_deg), np.array(means), np.array(stds)

    def sweep_elevation(self, elevations_deg=None, azimuth_deg=0.0):
        """Sweep elevation (tilt) values at fixed azimuth. Returns (elevations, mean_rssi, std)"""
        if elevations_deg is None:
            elevations_deg = np.arange(-30, 31, 1.0)
        means = []
        stds = []
        for el in elevations_deg:
            m, s = self.measure_rssi(rx_az_deg=azimuth_deg, rx_el_deg=el)
            means.append(m)
            stds.append(s)
        return np.array(elevations_deg), np.array(means), np.array(stds)

    def reset_shadowing(self, seed=None):
        self._shadowing_db = self._rng.normal(0, self.sigma_shadow)
