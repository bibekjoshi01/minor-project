import time
import numpy as np
from typing import Tuple, Dict, Any

from q_learning.agent import RLAgentBase
from q_learning.rl_env import RLEnvironment


def test_agent(
    agent: RLAgentBase,
    env_wrapper: RLEnvironment,
    max_steps: int = 200,
    samples: int = 1,
) -> Tuple[Dict[str, Any], list]:
    """
    Test a trained RL agent in the given environment.

    Returns:
        log_data: dictionary with metrics and parameters
        path: list of steps for plotting
    """

    state = env_wrapper.reset()
    total_reward = 0
    path = []

    start_time = time.time()

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done = env_wrapper.step(action)
        total_reward += reward

        # log current step
        path.append(
            {
                "pan": env_wrapper.pan,
                "tilt": env_wrapper.tilt,
                "rssi": env_wrapper.prev_rssi,
                "action": action,
            }
        )

        if done:
            break

        state = next_state

    end_time = time.time()
    convergence_time = end_time - start_time

    # extract best RSSI and location
    rssi_values = [p["rssi"] for p in path]
    best_idx = int(np.argmax(rssi_values))
    best_rssi = rssi_values[best_idx]
    best_point = {"pan": path[best_idx]["pan"], "tilt": path[best_idx]["tilt"]}

    log_data = {
        "algorithm": "RL Q-learning",
        "params": {
            "max_steps": max_steps,
        },
        "metrics": {
            "best_rssi": best_rssi,
            "best_point": best_point,
            "convergence_time_sec": convergence_time,
            "num_steps_taken": len(path),
            "total_samples": len(path) * samples,
        },
    }

    return log_data, path
