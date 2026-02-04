import numpy as np
import time


class StochasticHillClimb2D:
    """
    2D stochastic hill climbing for RSSI maximization.
    Adds randomness to avoid local maxima.
    """

    def __init__(
        self,
        env,
        pan_range=(30, 150),
        tilt_range=(60, 120),
        step_size=5,
        samples_per_point=10,
        max_iters=200,
        settle_time=0.0,
        init_point=None,
        patience=10,
        random_jump_prob=0.1,  # probability to jump randomly
    ):
        self.env = env

        self.pan_min, self.pan_max = pan_range
        self.tilt_min, self.tilt_max = tilt_range

        self.step = step_size
        self.samples = samples_per_point
        self.max_iters = max_iters
        self.settle_time = settle_time
        self.patience = patience
        self.random_jump_prob = random_jump_prob

        self.init_point = init_point
        self.scan_point = []

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

        self.scan_point.append(best_point.copy())
        no_improve = 0

        # ---- climb ----
        for _ in range(self.max_iters):
            # 8-neighborhood
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

            np.random.shuffle(neighbors)

            improved = False

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
                    self.scan_point.append(best_point.copy())
                    improved = True
                    break  # greedy ascent among shuffled neighbors

            # Random jump to escape local maxima
            if not improved and np.random.rand() < self.random_jump_prob:
                pan = np.random.randint(self.pan_min, self.pan_max + 1)
                tilt = np.random.randint(self.tilt_min, self.tilt_max + 1)
                self.env.set_orientation(pan, tilt)
                best_rssi = self.env.measure_rssi(self.samples)
                best_point = {"pan": pan, "tilt": tilt, "rssi": best_rssi}
                self.scan_point.append(best_point.copy())
                improved = True
                no_improve = 0

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= self.patience:
                break

        return best_rssi, best_point, self.scan_point


import math
import random
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

Number = float
Orient = Tuple[int, int]  # (pan, tilt)


def clamp(x, a, b):
    return max(a, min(b, x))


class StochasticHillClimber:
    """
    Hybrid hill-climber with simulated annealing, adaptive step size, averaging, tabu and restarts.

    env must implement:
      env.set_orientation(pan:int, tilt:int)
      env.measure_rssi() -> float

    Pan/Tilt are integers (degrees) aligned to step_deg.
    """

    def __init__(
        self,
        env,
        pan_min: int,
        pan_max: int,
        tilt_min: int,
        tilt_max: int,
        step_deg: int = 5,
        measure_avg_n: int = 3,
        init_temp: float = 1.0,
        temp_decay: float = 0.995,
        min_step_deg: int = 1,
        patience_reduce_step: int = 8,
        tabu_size: int = 50,
        restart_after: int = 200,
        random_restart_radius: int = 30,
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.env = env
        self.pan_min, self.pan_max = pan_min, pan_max
        self.tilt_min, self.tilt_max = tilt_min, tilt_max
        self.step_deg = step_deg
        self.measure_avg_n = max(1, measure_avg_n)
        self.temp = init_temp
        self.temp_decay = temp_decay
        self.min_step_deg = min_step_deg
        self.patience_reduce_step = patience_reduce_step
        self.tabu: Deque[Orient] = deque(maxlen=tabu_size)
        self.restart_after = restart_after
        self.random_restart_radius = random_restart_radius
        self.verbose = verbose

        # runtime state
        self.global_best: Orient = (pan_min, tilt_min)
        self.global_best_rssi = -1e9
        self.iter_since_improve = 0
        self.iter_total = 0

    # ----------------- helpers -----------------
    def _set_orientation(self, pan: int, tilt: int):
        pan = int(clamp(pan, self.pan_min, self.pan_max))
        tilt = int(clamp(tilt, self.tilt_min, self.tilt_max))
        self.env.set_orientation(pan, tilt)
        return pan, tilt

    def _measure_avg(self) -> float:
        # average multiple readings to reduce noise
        vals = []
        for _ in range(self.measure_avg_n):
            vals.append(self.env.measure_rssi())
        return float(sum(vals) / len(vals))

    def _grid_neighbors(self, pan: int, tilt: int, step: int) -> List[Orient]:
        # 8-neighborhood (plus cardinal)
        deltas = [
            (step, 0),
            (-step, 0),
            (0, step),
            (0, -step),
            (step, step),
            (step, -step),
            (-step, step),
            (-step, -step),
        ]
        neigh = []
        for dp, dt in deltas:
            p = clamp(pan + dp, self.pan_min, self.pan_max)
            t = clamp(tilt + dt, self.tilt_min, self.tilt_max)
            orient = (int(p), int(t))
            if orient not in neigh:
                neigh.append(orient)
        return neigh

    def _accept_worse(self, delta: float) -> bool:
        # delta = candidate_rssi - current_rssi (negative if worse)
        if delta >= 0:
            return True
        # simulated annealing acceptance
        prob = math.exp(delta / max(self.temp, 1e-9))
        return random.random() < prob

    # ----------------- main routine -----------------
    def run(
        self,
        max_iters: int = 2000,
        start_orient: Optional[Orient] = None,
        do_random_start: bool = True,
    ) -> Dict:
        """
        Returns a dict with best orientation and history.
        """
        # initialize start
        if start_orient is None or do_random_start:
            pan = random.randrange(self.pan_min, self.pan_max + 1, self.step_deg)
            tilt = random.randrange(self.tilt_min, self.tilt_max + 1, self.step_deg)
        else:
            pan, tilt = start_orient

        pan, tilt = self._set_orientation(pan, tilt)
        current_rssi = self._measure_avg()
        best_orient = (pan, tilt)
        best_rssi = current_rssi

        history = []
        step = self.step_deg
        no_improve_counter = 0

        for it in range(max_iters):
            self.iter_total += 1
            # generate neighbors
            neighbors = self._grid_neighbors(pan, tilt, step)

            # probe neighbors and pick best candidate (sample average each)
            candidates: List[Tuple[Orient, float]] = []
            for orient in neighbors:
                if orient in self.tabu:
                    # optional: skip recently visited
                    continue
                self._set_orientation(*orient)
                r = self._measure_avg()
                candidates.append((orient, r))

            # include staying as a candidate (helps preserve)
            candidates.append(((pan, tilt), current_rssi))

            # pick top candidate by measured RSSI
            candidates.sort(key=lambda x: x[1], reverse=True)
            cand_orient, cand_rssi = candidates[0]

            delta = cand_rssi - current_rssi

            accepted = self._accept_worse(delta)

            if accepted:
                # move to candidate
                pan, tilt = self._set_orientation(*cand_orient)
                current_rssi = cand_rssi
                self.tabu.append((pan, tilt))
                no_improve_counter = 0
            else:
                # didn't accept — stay
                no_improve_counter += 1

            # record history
            history.append(
                {
                    "iter": it,
                    "pan": pan,
                    "tilt": tilt,
                    "rssi": current_rssi,
                    "step": step,
                    "temp": self.temp,
                }
            )

            # update global best
            if current_rssi > best_rssi:
                best_rssi = current_rssi
                best_orient = (pan, tilt)
                # update global best too
                if best_rssi > self.global_best_rssi:
                    self.global_best_rssi = best_rssi
                    self.global_best = best_orient
                no_improve_counter = 0
                self.iter_since_improve = 0
            else:
                self.iter_since_improve += 1

            # anneal temperature
            self.temp *= self.temp_decay

            # reduce step size if stuck locally
            if (
                no_improve_counter >= self.patience_reduce_step
                and step > self.min_step_deg
            ):
                old_step = step
                step = max(self.min_step_deg, step // 2)
                no_improve_counter = 0
                if self.verbose:
                    print(f"[iter {it}] reducing step {old_step} -> {step}")

            # random restart to escape persistent local maxima
            if (it > 0 and it % self.restart_after == 0) or (
                self.iter_since_improve > 4 * self.restart_after
            ):
                # guided restart near best so far or fully random
                if random.random() < 0.7:
                    # near global best
                    rng = self.random_restart_radius
                    gp, gt = self.global_best
                    pan = int(
                        clamp(
                            gp + random.randint(-rng, rng), self.pan_min, self.pan_max
                        )
                    )
                    tilt = int(
                        clamp(
                            gt + random.randint(-rng, rng), self.tilt_min, self.tilt_max
                        )
                    )
                    if self.verbose:
                        print(
                            f"[iter {it}] guided restart near global best {self.global_best} rssi={self.global_best_rssi:.2f}"
                        )
                else:
                    pan = random.randrange(self.pan_min, self.pan_max + 1, step)
                    tilt = random.randrange(self.tilt_min, self.tilt_max + 1, step)
                    if self.verbose:
                        print(f"[iter {it}] random restart")
                pan, tilt = self._set_orientation(pan, tilt)
                current_rssi = self._measure_avg()
                self.tabu.clear()
                no_improve_counter = 0
                step = self.step_deg  # optionally reset to coarse
                continue

            # termination: if step is minimal and no improvement for long time
            if (
                step == self.min_step_deg
                and self.iter_since_improve >= 3 * self.restart_after
            ):
                if self.verbose:
                    print(
                        f"[iter {it}] no improvement for long time at min step; stopping."
                    )
                break

        # make sure we return the best measured point (evaluate global_best)
        self._set_orientation(*self.global_best)
        final_best_rssi = self._measure_avg()
        return {
            "best_orient": self.global_best,
            "best_rssi": final_best_rssi,
            "history": history,
        }
