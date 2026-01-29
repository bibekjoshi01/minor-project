import numpy as np

from base import AntennaEnvironmentBase, RLEnvironmentBase
from base import RLConfig, delta_rssi_sign, encode_state


class RLEnvironment(RLEnvironmentBase):
    def __init__(self, antenna_env: AntennaEnvironmentBase):
        """
        antenna_env = existing AntennaEnvironment (Simulation or Real)
        """

        self.env = antenna_env
        self.prev_rssi = None
        self.pan = None
        self.tilt = None

    def reset(self):
        # random discrete step aligned start
        pan_steps = (RLConfig.PAN_MAX - RLConfig.PAN_MIN) // RLConfig.STEP_DEG + 1
        tilt_steps = (RLConfig.TILT_MAX - RLConfig.TILT_MIN) // RLConfig.STEP_DEG + 1

        self.pan = (
            np.random.randint(0, pan_steps) * RLConfig.STEP_DEG + RLConfig.PAN_MIN
        )
        self.tilt = (
            np.random.randint(0, tilt_steps) * RLConfig.STEP_DEG + RLConfig.TILT_MIN
        )

        self.env.set_orientation(self.pan, self.tilt)
        rssi = self.env.measure_rssi()
        self.prev_rssi = rssi

        state = encode_state(self.pan, self.tilt, 0)
        return state

    def step(self, action_id):
        axis, direction = RLConfig.ACTIONS[action_id]

        # ----- apply action ------
        if axis == "PAN":
            self.pan += direction * RLConfig.STEP_DEG
        elif axis == "TILT":
            self.tilt += direction * RLConfig.STEP_DEG
        elif axis == "STAY":
            pass

        # ---- enforce mechanical limits ----
        self.pan = int(np.clip(self.pan, RLConfig.PAN_MIN, RLConfig.PAN_MAX))
        self.tilt = int(np.clip(self.tilt, RLConfig.TILT_MIN, RLConfig.TILT_MAX))

        # ---- apply to system ----
        self.env.set_orientation(self.pan, self.tilt)
        rssi = self.env.measure_rssi()

        # ---- reward logic ----
        delta = rssi - self.prev_rssi
        delta_sign = delta_rssi_sign(delta)
        reward = delta

        state = encode_state(self.pan, self.tilt, delta_sign)
        self.prev_rssi = rssi
        done = False
        return state, reward, done
