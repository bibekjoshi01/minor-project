import numpy as np
import pickle

from .base import RLAgentBase
from .config import RLConfig


class QLearningAgent(RLAgentBase):
    def __init__(
        self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995
    ):
        # Q-table: pan x tilt x delta_rssi_sign x actions
        pan_states = (RLConfig.PAN_MAX - RLConfig.PAN_MIN) // RLConfig.STEP_DEG + 1
        tilt_states = (RLConfig.TILT_MAX - RLConfig.TILT_MIN) // RLConfig.STEP_DEG + 1
        delta_states = 3
        n_actions = len(RLConfig.ACTIONS)

        self.q_table = np.zeros(
            (pan_states, tilt_states, delta_states, n_actions), dtype=np.float32
        )

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # ---- Action selection ----
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(RLConfig.ACTIONS))
        else:
            pan_i, tilt_i, delta_i = state
            return int(np.argmax(self.q_table[pan_i, tilt_i, delta_i]))

    # ---- Q-table update ----
    def update(self, state, action, reward, next_state):
        pan_i, tilt_i, delta_i = state
        pan_n, tilt_n, delta_n = next_state

        best_next = np.max(self.q_table[pan_n, tilt_n, delta_n])

        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[pan_i, tilt_i, delta_i, action]

        self.q_table[pan_i, tilt_i, delta_i, action] += self.alpha * td_error

    # ---- Exploration decay ----
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ---- Save / Load ----
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)
