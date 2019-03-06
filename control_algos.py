import numpy as np
from numpy.random import multinomial


class Sarsa:
    def __init__(self, discount, n_states, n_actions, temp, learning_rate):
        self.learning_rate = learning_rate
        self.temp = temp
        self.discount = discount
        self.q = np.zeros((n_states, n_actions))

    def __str__(self):
        return "Sarsa"

    def backup_state_action_values(self, s, a, r, next_s, next_a):
        self.q[s, a] = self.q[s, a] + self.learning_rate * (r + self.discount * self.q[next_s, next_a] - self.q[s, a])
        return self.q[s, a]

    def sample_action(self, state):
        action_probabilities = get_action_probabilities(self.q[state], self.temp)
        return np.argmax(multinomial(1, action_probabilities))

    def optimal_policy(self, s):
        return np.argmax(self.q[s])


class ExpectedSarsa:
    def __init__(self, discount, n_states, n_actions, temp, learning_rate):
        self.learning_rate = learning_rate
        self.temp = temp
        self.discount = discount
        self.q = np.zeros((n_states, n_actions))

    def backup_state_action_values(self, s, a, r, next_s, *args):
        action_probabilities = np.array(get_action_probabilities(self.q[s], self.temp))
        bootstrap_target = action_probabilities.dot(self.q[next_s])
        self.q[s, a] = self.q[s, a] + self.learning_rate * (r + self.discount * bootstrap_target - self.q[s, a])
        return self.q[s, a]

    def sample_action(self, state):
        action_probabilities = get_action_probabilities(self.q[state], self.temp)
        return np.argmax(multinomial(1, action_probabilities))

    def optimal_policy(self, s):
        return np.argmax(self.q[s])


class QLearning:
    def __init__(self, discount, n_states, n_actions, temp, learning_rate):
        self.learning_rate = learning_rate
        self.temp = temp
        self.discount = discount
        self.q = np.zeros((n_states, n_actions))

    def backup_state_action_values(self, s, a, r, next_s, *args):
        next_a = self.optimal_policy(s)
        self.q[s, a] = self.q[s, a] + self.learning_rate * (r + self.discount * self.q[next_s, next_a] - self.q[s, a])
        return self.q[s, a]

    def sample_action(self, state):
        action_probabilities = get_action_probabilities(self.q[state], self.temp)
        return np.argmax(multinomial(1, action_probabilities))

    def optimal_policy(self, s):
        return np.argmax(self.q[s])


def get_action_probabilities(action_values, temp):
    nums = np.exp(action_values / temp)
    normalizer = np.sum(nums)
    return [num / normalizer for num in nums]
