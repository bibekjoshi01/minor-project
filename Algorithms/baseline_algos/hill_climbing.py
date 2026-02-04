import numpy as np
import time

class HillClimb2D:
    """
    2D hill climbing (pan, tilt) for RSSI maximization.
    Local optimizer, not global.
    """

    def __init__(
        self,
        env,
        pan_range=(0, 360),
        tilt_range=(60, 120),
        step_size=5,  # angular step
        samples_per_point=10,
        max_iters=200,
        settle_time=0.0,
        init_point=None,
        patience=10,  # stop if no improvement
    ):
        self.env = env

        self.pan_min, self.pan_max = pan_range
        self.tilt_min, self.tilt_max = tilt_range
        self.step = step_size

        self.samples = samples_per_point
        self.settle_time = settle_time
        self.max_iters = max_iters
        self.patience = patience

        self.init_point = init_point

        self.scan_data = []  # trajectory of points

    # ---------------- Core ----------------
    def run_scan(self):
        if self.init_point is None:
            pan = np.random.randint(self.pan_min, self.pan_max + 1)
            tilt = np.random.randint(self.tilt_min, self.tilt_max + 1)
        else:
            pan, tilt = self.init_point

        self.env.set_orientation(pan, tilt)
        best_rssi = self.env.measure_rssi(self.samples)
        best_point = {"pan": pan, "tilt": tilt, "rssi": best_rssi}

        self.scan_data.append(best_point.copy())

        no_improve = 0

        # ---- climb ----
        for _ in range(self.max_iters):

            improved = False

            # 8-neighborhood search
            neighbors = [
                (pan + self.step, tilt),
                (pan - self.step, tilt),
                (pan, tilt + self.step),
                (pan, tilt - self.step),
                (pan + self.step, tilt + self.step),
                (pan + self.step, tilt - self.step),
                (pan - self.step, tilt + self.step),
                (pan - self.step, tilt - self.step),
            ]

            for p, t in neighbors:
                p = int(np.clip(p, self.pan_min, self.pan_max))
                t = int(np.clip(t, self.tilt_min, self.tilt_max))

                self.env.set_orientation(p, t)
                rssi = self.env.measure_rssi(self.samples)

                if self.settle_time > 0:
                    time.sleep(self.settle_time)

                if rssi > best_rssi:
                    best_rssi = rssi
                    pan, tilt = p, t
                    best_point = {"pan": pan, "tilt": tilt, "rssi": best_rssi}
                    self.scan_data.append(best_point.copy())
                    improved = True
                    break  # greedy ascent

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= self.patience:
                break

        return best_rssi, best_point, self.scan_data
