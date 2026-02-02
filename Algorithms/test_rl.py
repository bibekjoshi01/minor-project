from q_learning.agent import QLearningAgent
from q_learning.rl_env import RLEnvironment
from q_learning.antenna_env import AntennaEnvironmentSim
from q_learning.train import train_agent

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
