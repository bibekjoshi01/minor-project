import numpy as np
import time

from antenna_environment import AntennaEnvironment


class ExhaustiveScan2D:
    """
    Exhaustive scan over pan (azimuth) and tilt (elevation).
    Designed for simulation first, hardware later.
    """

    def __init__(
        self,
        env,
        pan_range=(0, 360),
        tilt_range=(60, 120),  # Assuming 90 as horizontal
        pan_step=10,
        tilt_step=5,
        fine_step=2,
        samples_per_point=10,
        settle_time=0.0,  # simulation delay in seconds (0 for sim, >0 for hardware)
    ):
        self.env: AntennaEnvironment = env

        self.pan_min, self.pan_max = pan_range
        self.pan_step = pan_step

        self.tilt_min, self.tilt_max = tilt_range
        self.tilt_step = tilt_step

        self.samples = samples_per_point
        self.settle_time = settle_time
        self.fine_step = fine_step

        self.scan_data = []  # stores dicts: {pan, tilt, rssi}

    # ---------------- Scan ----------------
    def _run_fine_scan(self, coarse_pan, coarse_tilt):
        """
        Local scan around coarse optimum
        """

        best_rssi = -1e9
        best_point = None
        fine_data = []

        for pan in range(coarse_pan - 10, coarse_pan + 11, self.fine_step):
            for tilt in range(coarse_tilt - 10, coarse_tilt + 11, self.fine_step):

                # clip angles to valid range
                pan = np.clip(pan, self.pan_min, self.pan_max)
                tilt = np.clip(tilt, self.tilt_min, self.tilt_max)

                self.env.set_orientation(pan, tilt)
                rssi = self.env.measure_rssi(self.samples)

                if self.settle_time > 0:
                    time.sleep(self.settle_time)

                point = {"pan": pan, "tilt": tilt, "rssi": rssi}
                fine_data.append(point)

                if rssi > best_rssi:
                    best_rssi = rssi
                    best_point = point.copy()

        return best_rssi, best_point, fine_data

    def run_scan(self) -> tuple[float, dict, list[dict]]:
        """
        Performs full 2D exhaustive scan.
        Returns:
            best_rssi: number
            best_point: dict {pan, tilt, rssi}
            scan_data: list of all measured points
        """

        self.scan_data = []
        best_rssi = -1e9
        best_point = None

        for pan in range(self.pan_min, self.pan_max + 1, self.pan_step):
            for tilt in range(self.tilt_min, self.tilt_max + 1, self.tilt_step):

                self.env.set_orientation(pan, tilt)
                rssi = self.env.measure_rssi(self.samples)

                if self.settle_time > 0:
                    time.sleep(self.settle_time)

                point = {"pan": pan, "tilt": tilt, "rssi": rssi}

                self.scan_data.append(point)

                if rssi > best_rssi:
                    best_rssi = rssi
                    best_point = point.copy()

        best_rssi, best_point, fine_data = self._run_fine_scan(
            best_point["pan"], best_point["tilt"]
        )
        self.scan_data.extend(fine_data)

        return best_rssi, best_point

    # ---------------- Utilities ----------------
    def get_rssi_grid(self):
        """
        Converts scan_data to 2D grid for plotting/analysis.
        Returns:
            pan_vals: 1D array of pan angles
            tilt_vals: 1D array of tilt angles
            rssi_grid: 2D array of RSSI values (shape: len(pan_vals) x len(tilt_vals))
        """
        pan_vals = np.unique([p["pan"] for p in self.scan_data])
        tilt_vals = np.unique([p["tilt"] for p in self.scan_data])

        grid = np.zeros((len(pan_vals), len(tilt_vals)))

        # lookup dicts for fast indexing
        pan_idx = {angle: i for i, angle in enumerate(pan_vals)}
        tilt_idx = {angle: i for i, angle in enumerate(tilt_vals)}

        # Fill grid
        for p in self.scan_data:
            i = pan_idx[p["pan"]]
            j = tilt_idx[p["tilt"]]
            grid[i, j] = p["rssi"]

        return pan_vals, tilt_vals, grid
