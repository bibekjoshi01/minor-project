from .base import RLEnvironmentBase, RLAgentBase
import numpy as np


def train_agent(
    agent: RLAgentBase,
    env_wrapper: RLEnvironmentBase,
    n_episodes=500,
    max_steps=100,
    patience=10,
    eps_stable=0.1,
    save_path="q_table.npy",
):
    """
    Trains any RL agent (inherits RLAgentBase) on the environment.
    """
    history = []
    global_best_rssi = -1e9

    for ep in range(n_episodes):
        state = env_wrapper.reset()
        total_reward = 0
        steps_no_improve = 0
        best_rssi = -1e9

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done = env_wrapper.step(action)

            # clip reward to reduce noise impact
            reward = np.clip(reward, -5, 5)

            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # stabilization check
            if reward <= eps_stable:
                steps_no_improve += 1
            else:
                steps_no_improve = 0

            rssi_now = env_wrapper.prev_rssi
            if rssi_now > best_rssi:
                best_rssi = rssi_now

            if steps_no_improve >= patience or done:
                break

        agent.decay_epsilon()
        history.append(total_reward)

        if best_rssi > global_best_rssi:
            global_best_rssi = best_rssi

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{n_episodes} | Total Reward: {total_reward:.2f} |"
                f"Best RSSI: {best_rssi:.2f} | Global Best RSSI: {global_best_rssi:.2f} |"
                f"ε={agent.epsilon:.3f}"
            )

    # final save
    agent.save(save_path)
    print(f"Q-table saved to {save_path}")

    return history
