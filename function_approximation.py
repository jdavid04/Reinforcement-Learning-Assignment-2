import numpy as np
import gym
from tqdm import tqdm


class PendulumTD:
    """
    Implementation of TD-lambda for the Pendulum environment of OpenAI Gym.

    Assumes linear function approximation.
    """

    def __init__(self, num_bins, x_low, x_high, y_low, y_high, discount, decay,
                 learning_rate, num_tilings=5, seed=0, offset=0.03, tile_offsets=None):
        """
        Initialize an object to perform TD-lambda policy evaluation for the given fixed policy.

        :param num_bins: number of bins to split each dimension of state space into.
        :param x_low: min value of x-axis dimension.
        :param x_high: max value of x-axis dimension.
        :param y_low: min value of y-axis dimension.
        :param y_high: max value of y-axis dimension.
        :param discount: gamma discount factor.
        :param decay: lambda decay for eligibility traces.
        :param learning_rate: learning rate for TD-lambda.
        :param num_tilings: number of tilings to tile coding.
        """

        np.random.seed(seed)
        self.offset = offset
        if tile_offsets is None:
            self.tile_offsets = np.array([
                [-offset, offset, -offset, offset],
                [0, 2 * offset, -offset, offset],
                [-offset, offset, 0, 2 * offset],
                [-2. * offset, 0, -offset, offset],
                [-offset, offset, -2. * offset, 0]])
        else:
            self.tile_offsets = tile_offsets
        self.y_high = y_high
        self.y_low = y_low
        self.x_high = x_high
        self.x_low = x_low
        self.x_diff = (x_high - x_low) * (1 + 2 * offset)
        self.y_diff = (y_high - y_low) * (1 + 2 * offset)
        self.num_features = (num_bins ** 2) * num_tilings
        self.num_tilings = num_tilings
        self.num_bins = num_bins
        self.weights = np.random.uniform(low=-0.001, high=.001, size=self.num_features)
        self.traces = np.zeros(self.num_features)
        self.discount = discount  # Corresponds to gamma discount factor
        self.decay = decay  # Corresponds to lambda decay factor
        self.learning_rate = learning_rate / self.num_tilings

    def update(self, state, reward, next_state):
        """
        Perform the TD-lambda algorithm backup step. Update the object's weight vector in-place.

        :param state: state as given by environment.
        :param reward: observed reward upon taking action in state.
        :param next_state: sample next state in feature vector form.
        :return: None
        """

        state = self.get_features(self._get_observation_state(state))
        next_state = self.get_features(self._get_observation_state(next_state))
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

    def get_features(self, state):
        """
        Transform the given observation into a feature vector description by tile coding.

        :param list observation:  current state, given by theta, thetadot
        :return: feature vector of length 500 (5 tilings of 100 tiles), with 5 indices set to 1 for each observation
        """
        theta, thetadot = state
        features = np.zeros(self.num_features)
        features[self._compute_active_tiles(theta, thetadot)] = 1.
        return features

    def _compute_active_tiles(self, x, y):
        """
        Input the state coordinates and compute the active tile indices (0-499).

        :param x: theta angular position.
        :param y: thetadot angular velocity.
        :return: list of 5 active tile indices.
        """

        tile_indices = []
        for i in range(self.num_tilings):
            offsets = self.tile_offsets[i]
            x_high = self.x_high + offsets[1]
            y_high = self.y_high + offsets[3]

            x = x if x >= 0 else x + x_high
            y = y if y >= 0 else y + y_high
            x_coord = x // (self.x_diff / self.num_bins)
            y_coord = y // (self.y_diff / self.num_bins)
            idx_multiplier = i * self.num_bins ** 2
            tile_indices.append(int(y_coord * self.num_bins + x_coord) + idx_multiplier)
        return tile_indices

    def _get_observation_state(self, observation):
        """
        Obtain true state from observation.

        :param observation: observation as returned by the environment (cos theta, sin theta, thetadot).
        :return: theta, thetadot state representation.
        """
        theta = np.arccos(observation[0])
        theta = theta if observation[1] >= 0 else -theta
        thetadot = observation[2]
        return theta, thetadot

    def pendulum_fixed_policy(self, thetadot):
        """
        Apply the fixed policy from assignment to generate next action.

        :param float thetadot: angular velocity of the pendulum.
        :return: torque to be applied, +2 or -2.
        """
        p = [0.9, 0.1] if thetadot > 0 else [0.1, 0.9]
        return [np.random.choice([2., -2.], p=p)]


def run(num_episodes, pendulum_algo, env):
    for _ in tqdm(range(num_episodes)):
        run_episode(env, pendulum_algo)
    return pendulum_algo.function_approximation(state=pendulum_algo.get_features((0, 0)))


def run_episode(env, pendulum_algo):
    env.reset()
    s = _initialize_episode(env)
    a = pendulum_algo.pendulum_fixed_policy(thetadot=s[2])
    done = False
    while not done:
        observation, reward, done, _ = env.step(a)
        next_a = pendulum_algo.pendulum_fixed_policy(observation[2])
        pendulum_algo.update(state=s, reward=reward, next_state=observation)
        s, a = observation, next_a
    return pendulum_algo.function_approximation(state=pendulum_algo.get_features((0, 0)))


def _initialize_episode(env, start_state=[0, 0]):
    env.unwrapped.state = np.array(start_state)
    return np.array([1., 0., 0.])


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    lambd = 1
    learning_rate = 0.25
    pendulum_algo = PendulumTD(num_bins=10, x_low=-np.pi, x_high=np.pi, y_low=-8., y_high=8.,
                               discount=0.9, decay=lambd,
                               learning_rate=learning_rate,
                               num_tilings=5, seed=0)
    run_episode(env, pendulum_algo)
