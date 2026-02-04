import numpy as np
from typing import Tuple, Optional

from .base import RLAgentBase
from .config import RLConfig


class QLearningTrainer(RLAgentBase):
    """
    Offline Q-learning trainer for antenna orientation.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.05,  # low gamma = static optimization
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 1.0,
        seed: Optional[int] = None,
        q_clip: Tuple[float, float] = (-10.0, 10.0),
    ):
        if seed is not None:
            np.random.seed(seed)

        # State space
        self.pan_states = (RLConfig.PAN_MAX - RLConfig.PAN_MIN) // RLConfig.STEP_DEG + 1
        self.tilt_states = (
            RLConfig.TILT_MAX - RLConfig.TILT_MIN
        ) // RLConfig.STEP_DEG + 1
        self.delta_states = 3  # {-1, 0, +1}

        if self.pan_states <= 0 or self.tilt_states <= 0:
            raise ValueError("Invalid PAN/TILT configuration")

        # Actions
        self.n_actions = len(RLConfig.ACTIONS)

        # Q-table
        self.Q = np.full(
            (
                self.pan_states,
                self.tilt_states,
                self.delta_states,
                self.n_actions,
            ),
            optimistic_init,
            dtype=np.float32,
        )

        # Learning params
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        # Exploration
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Safety
        self.q_clip = q_clip

    # Internal safety
    def _check_state(self, s: Tuple[int, int, int]):
        p, t, d = s
        if not (0 <= p < self.pan_states):
            raise ValueError(f"Pan index out of range: {p}")
        if not (0 <= t < self.tilt_states):
            raise ValueError(f"Tilt index out of range: {t}")
        if not (0 <= d < self.delta_states):
            raise ValueError(f"Delta index out of range: {d}")

    # Action selection (ε-greedy with tie-breaking)
    def select_action(self, state: Tuple[int, int, int]) -> int:
        self._check_state(state)
        p, t, d = state

        # exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # greedy with random tie-breaking
        q = self.Q[p, t, d]
        max_q = np.max(q)
        best_actions = np.flatnonzero(q == max_q)

        return int(np.random.choice(best_actions))

    # Q-learning update
    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
    ):
        self._check_state(state)
        self._check_state(next_state)

        p, t, d = state
        pn, tn, dn = next_state

        best_next = np.max(self.Q[pn, tn, dn])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[p, t, d, action]

        self.Q[p, t, d, action] += self.alpha * td_error

        # clip for numerical stability
        self.Q[p, t, d, action] = np.clip(
            self.Q[p, t, d, action],
            self.q_clip[0],
            self.q_clip[1],
        )

    # Epsilon decay
    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # Export
    def save(self, path: str):
        """Save the raw Q-table to a .npy file."""
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, self.Q)
        print(f"Q-table saved to '{path}' with shape {self.Q.shape}")

    # Diagnostics
    def q_stats(self):
        return {
            "mean": float(np.mean(self.Q)),
            "std": float(np.std(self.Q)),
            "max": float(np.max(self.Q)),
            "min": float(np.min(self.Q)),
        }
