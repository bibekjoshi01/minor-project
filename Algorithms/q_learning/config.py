class RLConfig:
    PAN_MIN = 0
    PAN_MAX = 350

    TILT_MIN = 60
    TILT_MAX = 120

    STEP_DEG = 5

    ACTIONS = {
        0: ("PAN", +1),
        1: ("PAN", -1),
        2: ("TILT", +1),
        3: ("TILT", -1),
        4: ("STAY", 0),
    }

    DELTA_RSSI_MAP = {-1: 0, 0: 1, +1: 2}  # decrease  # no change  # increase


def discretize_angle(angle, min_a, step):
    return int((angle - min_a) // step)


def delta_rssi_sign(delta, eps=0.1):
    if delta > eps:
        return +1
    elif delta < -eps:
        return -1
    else:
        return 0


def encode_state(pan, tilt, delta_rssi):
    pan_i = discretize_angle(pan, RLConfig.PAN_MIN, RLConfig.STEP_DEG)
    tilt_i = discretize_angle(tilt, RLConfig.TILT_MIN, RLConfig.STEP_DEG)
    d_i = RLConfig.DELTA_RSSI_MAP[delta_rssi]
    return (pan_i, tilt_i, d_i)
