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
        self.tiling = self._initialize_tiling(num_features)

    def _initialize_tiling(self, num_features):
        pass  # TODO

    def update(self, state, reward, next_state):
        """
        Perform the TD-lambda algorithm backup step. Update the object's weight vector in-place.

        :param state: feature vector of state given by get_features.
        :param reward: observed reward upon taking action in state.
        :param next_state: sample next state in feature vector form.
        :return: None
        """
        self.traces = self.discount * self.decay * self.traces + state
        delta = reward + self.discount * self.function_approximation(next_state) - self.function_approximation(state)
        self.weights = self.weights + self.learning_rate * delta * self.traces

    def function_approximation(self, state):
        """
        Compute state value by linear function approximation.

        :param state: feature vector representation of the current state.
        :return: predicted state value computed by linear function approximation.
        """
        return self.weights.dot(state)

    def get_features(self, observation):
        """
        Transform the given observation into a feature vector description by tile coding.

        :param observation:  current state, given by theta, thetadot
        :return: feature vector of length 500 (5 tilings of 100 tiles), with 5 set to 1 for each observation
        """
        theta, thetadot = self._get_observation_state(observation)
        feature_vector = self._compute_active_tiles(theta, thetadot)

        return feature_vector

    def _compute_active_tiles(self, x, y):
        pass  # TODO

    def _get_observation_state(self, observation):
        """

        :param observation: observation as returned by the environment (cos theta, sin theta, thetadot).
        :return: theta, thetadot state representation.
        """
        theta = np.arccos(observation[0])  # TODO check this
        thetadot = observation[2]
        return theta, thetadot

    def _pendulum_fixed_policy(self, thetadot):
        p = [0.9, 0.1] if thetadot > 0 else [0.1, 0.9]
        return np.random.choice([[2], [-2]], p=p)
