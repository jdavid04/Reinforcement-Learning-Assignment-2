import numpy as np

from utils import BaseAgent

from itertools import product


class Tiling:

    def __init__(self):

        self.state_bounds = np.array([np.pi, 8])

        full_range = self.state_bounds * 2
        self.inflated_range = full_range * 1.08

        offset = (self.inflated_range - full_range) / 2

        h, w = offset

        self.deltas = np.array(
            [[i * h, j * w] for i, j in product([-1, 1], [-1, 1])] + [[0, 0]]) + offset

    def __call__(self, observation):
        """
        Returns the feature vector associated to the observation.

        Args:
            observation (numpy.array): The observation made from the environment.

        Returns:
            feature (numpy.array): The feature vector associated to the provided observation).
        """

        cos, sin, thetadot = observation
        theta = np.arctan2(sin, cos)

        state = np.array([theta, thetadot])

        feature = np.zeros((5, 10, 10), dtype=int)

        for t, delta in enumerate(self.deltas):

            i, j = np.minimum(
                (state + self.state_bounds + delta) // (self.inflated_range / 10),
                9
            ).astype(int)

            feature[t, i, j] = 1

        feature = feature.reshape(-1)

        return feature


class Agent(BaseAgent):

    def __init__(self, environment, gamma=1, alpha=1/4, lambd=0, verbose=False):

        super(Agent, self).__init__(verbose)

        self.environment = environment

        # Get the action max
        self.action_max = environment.action_space.high[0]

        # Building the feature-vector
        self.weights = None
        self.eligibility = None

        # Parameter of the agent
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd

        # Useful quantities
        self.observation = None

        self.tiling = Tiling()

    def initialise(self, seed=None):
        """
        Initialises the agent (random seed, eligibility set to 0 and feature weights initialised).

        Args:
            seed (int): The random seed to use.
        """

        np.random.seed(seed)
        self.environment.seed(seed)
        self.weights = np.random.uniform(-.001, .001, 500)
        self.eligibility = np.zeros(500)

    def get_features(self, observation):
        r"""
        Returns the feature vector associated with a given observation.

        Since the observation does not provide $\theta$ directly, but rather $\cos \theta$ and $\sin \theta$,
        we need to transform those into an angle.

        Args:
            observation (np.array): The array representing the observation.

        Returns:
            features (np.array): The feature vector.
        """

        return self.tiling(observation)

    def policy(self, observation):
        """
        Returns the action taken according to the given policy:

            The fixed policy produces torque in the same direction as the current velocity
            with probability 0.9 and in the opposite direction with probability 0.1.

        Args:
            observation (np.array): The observation upon which the agen decides on the action to take.

        Returns:
            action (np.array): The torque to apply.
        """

        cos, sin, thetadot = observation
        action = np.sign(thetadot) * self.action_max

        # We choose the action according to the policy.
        action = np.random.choice([action, -action], p=[.9, .1])

        return np.array([action], dtype=np.float32)

    def backup(self):
        r"""
        Performs a single backup operation.

        Note:
            The backward-view TD(:math:`\lambda`) with linear function approximation is:

            .. math::

                \delta_t = r_{t+1} + \gamma \hat{v} (s_{t+1}, w) - \hat{v} (s_{t}, w)

                e_t = \gamma \lambda e_{t-1} + x(s_{t})

                \Delta w = \alpha \delta_t e_t
        """

        action = self.policy(self.observation)
        feature = self.get_features(self.observation)

        o, r, d, i = self.environment.step(action)

        new_feature = self.get_features(o)

        delta = r + self.gamma * self.weights @ new_feature - self.weights @ feature

        self.eligibility = self.gamma * self.lambd * self.eligibility + feature

        effective_alpha = self.alpha / feature.sum()

        self.weights += effective_alpha * self.eligibility * delta

        self.observation = o

        return d, r

    def reset_environment(self):
        """
        Resets the environment to state (0, 0).
        """

        self.environment.reset()
        self.environment.unwrapped.state = np.zeros(2)

        return np.array([1., 0., 0.])

    def episode(self):
        """
        Runs a full episode.

        Returns:
            full_return (float): The full return obtained during the experiment.
        """

        done = False
        full_return = 0.

        self.observation = self.reset_environment()

        while not done:
            done, reward = self.backup()
            full_return = self.gamma * full_return + reward

        state_value = self.weights @ self.get_features((1, 0, 0))

        return full_return, state_value

    def run(self, gamma=.9, alpha=1/4, lambd=0, seed=None, episodes=200):
        """
        Performs a full run:
        - initialises the agent ;
        - performs 200 episodes.

        Args:
            gamma (float): The discount factor.
            alpha (float): The learning rate.
            lambd (float): The eligibility trace factor.
            seed (int): The random seed to use.
            episodes (int): The number of episodes to run.

        Returns:
            start_value (float): The value of the first state (0, 0).
        """

        self.initialise(seed)

        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd

        state_values = np.empty(episodes)

        iterator = self.tqdm(range(episodes))

        for i in iterator:
            full_return, state_value = self.episode()
            state_values[i] = state_value

        return state_values


if __name__ == '__main__':

    tiling = Tiling()
    tiling((0, 1, 2))
