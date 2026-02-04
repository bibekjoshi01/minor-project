from q_learning.agent import QLearningTrainer
from q_learning.rl_env import RLEnvironment
from ant_env.environment import AntennaEnvironmentSim
from q_learning.train import train_agent

antenna_env = AntennaEnvironmentSim()
env_wrapper = RLEnvironment(antenna_env)
agent = QLearningTrainer()

history = train_agent(agent, env_wrapper, n_episodes=500, max_steps=100, patience=10)

# plotting
import matplotlib.pyplot as plt
import numpy as np

history = np.array(history)

# Compute cumulative max RSSI per episode
cumulative_best = np.maximum.accumulate(history)

# Moving average for smoothing
window = 10
moving_avg = np.convolve(history, np.ones(window) / window, mode="valid")

# ---- 1: Reward per episode ----
plt.figure(figsize=(10, 5))
plt.plot(history, label="Reward per episode", alpha=0.5)
plt.plot(
    range(window - 1, len(history)),
    moving_avg,
    label=f"Moving avg ({window})",
    color="orange",
)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Reward per Episode")
plt.legend()
plt.grid(True)
plt.show()

# ---- 2: Cumulative best RSSI ----
plt.figure(figsize=(10, 5))
plt.plot(cumulative_best, color="green")
plt.xlabel("Episode")
plt.ylabel("Cumulative Best Reward / RSSI")
plt.title("Cumulative Best RSSI Over Episodes")
plt.grid(True)
plt.show()

# ---- 3: Epsilon decay ----
# Assuming you saved epsilon per episode:
if hasattr(agent, "epsilon_history"):
    plt.figure(figsize=(10, 4))
    plt.plot(agent.epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Rate Decay")
    plt.grid(True)
    plt.show()

# ---- 4: Comparison with baselines ----
# Example numbers for comparison (replace with actual measurements)
baseline_rewards = {
    "RL": np.max(history),
    "Hill Climbing": 8.2,  # example
    "Exhaustive Search": 9.0,  # example
}

plt.figure(figsize=(6, 5))
plt.bar(
    baseline_rewards.keys(), baseline_rewards.values(), color=["blue", "red", "green"]
)
plt.ylabel("Max RSSI / Reward")
plt.title("RL vs Hill Climbing vs Exhaustive Search")
plt.show()
