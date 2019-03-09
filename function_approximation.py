import numpy as np


class PendulumTD:
    """
    Implementation of TD-lambda for the Pendulum environment of OpenAI Gym.

    Assumes linear function approximation.
    """

    def __init__(self, num_features, discount, decay, learning_rate):
        self.weights = np.zeros(num_features)
        self.traces = np.zeros(num_features)
        self.discount = discount  # Corresponds to gamma discount factor
        self.decay = decay  # Corresponds to lambda decay factor
        self.learning_rate = learning_rate

    def update(self, state, reward, next_state):
        self.traces = self.discount * self.decay * self.traces + state
        delta = reward + self.discount * self.function_approximation(next_state) - self.function_approximation(state)
        self.weights = self.weights + self.learning_rate * delta * self.traces

    def function_approximation(self, state):
        return self.weights.dot(state)

    def _pendulum_fixed_policy(self, theta, thetadot):
        pass
