from agent import QLearningAgent
from rl_env import RLEnvironment
from antenna_env import AntennaEnvironmentSim
from train import train_agent

antenna_env = AntennaEnvironmentSim()
env_wrapper = RLEnvironment(antenna_env)

agent = QLearningAgent()
history = train_agent(agent, env_wrapper, n_episodes=500, max_steps=100, patience=10)

# plotting
import matplotlib.pyplot as plt

plt.plot(history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Reward per Episode")
plt.grid(True)
plt.show()
