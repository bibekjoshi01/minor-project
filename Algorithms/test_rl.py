import pickle
import time
import math

MODEL_PATH = "q_table.pkl"

NUM_ACTIONS = 4
RSSI_MIN = -100
RSSI_MAX = -30
RSSI_BINS = 8

ANGLE_MIN = 0
ANGLE_MAX = 360
ANGLE_BINS = 8

STEP_DELAY_SEC = 0.1  # control loop delay

STATE_SPACE_SIZE = RSSI_BINS * ANGLE_BINS


# LOAD Q-TABLE (BOOT)
def load_q_table(path):
    with open(path, "rb") as f:
        q_table = pickle.load(f)

    if not isinstance(q_table, dict):
        raise TypeError("Q-table must be dict")

    for s, q in q_table.items():
        if len(q) != NUM_ACTIONS:
            raise ValueError("Action size mismatch")

    print(f"[BOOT] Q-table loaded ({len(q_table)} states)")
    return q_table


Q_TABLE = load_q_table(MODEL_PATH)


# STATE ENCODING (CRITICAL)
def discretize(value, vmin, vmax, bins):
    if value <= vmin:
        return 0
    if value >= vmax:
        return bins - 1
    return int((value - vmin) / (vmax - vmin) * bins)


def encode_state(rssi, angle):
    rssi_bin = discretize(rssi, RSSI_MIN, RSSI_MAX, RSSI_BINS)
    angle_bin = discretize(angle, ANGLE_MIN, ANGLE_MAX, ANGLE_BINS)
    return rssi_bin * ANGLE_BINS + angle_bin


# ACTION SELECTION (GREEDY)
def select_action(state):
    q_values = Q_TABLE.get(state)

    if q_values is None:
        return 0  # safe fallback

    best_action = 0
    best_value = q_values[0]

    for i in range(1, NUM_ACTIONS):
        if q_values[i] > best_value:
            best_value = q_values[i]
            best_action = i

    return best_action


# ACTUATION (STUB)
def execute_action(action):
    if action == 0:
        print("ACTION: ROTATE_CW")
    elif action == 1:
        print("ACTION: ROTATE_CCW")
    elif action == 2:
        print("ACTION: HOLD")
    elif action == 3:
        print("ACTION: FINE_ADJUST")


# SENSOR INPUT (SIMULATION ONLY)
def read_sensors():
    rssi = -65 + 4 * math.sin(time.time())
    angle = (time.time() * 20) % 360
    return rssi, angle


# MAIN LOOP
def main():
    print("[RUN] RL controller active")

    while True:
        rssi, angle = read_sensors()
        state = encode_state(rssi, angle)
        action = select_action(state)

        print(f"RSSI={rssi:.1f}  ANGLE={angle:.1f}  STATE={state}  ACTION={action}")
        execute_action(action)

        time.sleep(STEP_DELAY_SEC)


if __name__ == "__main__":
    main()
