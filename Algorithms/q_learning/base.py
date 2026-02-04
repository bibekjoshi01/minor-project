from abc import ABC, abstractmethod


class RLAgentBase(ABC):
    @abstractmethod
    def select_action(self, state):
        """Return action index for a given state"""
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state):
        """Update internal Q-table or policy"""
        pass

    @abstractmethod
    def decay_epsilon(self):
        """Decay exploration rate if applicable"""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save internal Q-table or model"""
        pass


class RLEnvironmentBase(ABC):
    @abstractmethod
    def reset(self, pan, tilt):
        pass

    @abstractmethod
    def step(self):
        pass
