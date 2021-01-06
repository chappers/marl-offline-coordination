import random

from marlkit.exploration_strategies.base import RawExplorationStrategy


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """

    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
        return action


class MAEpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """

    def __init__(self, action_space, n_agents=1, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space
        self.n_agents = n_agents

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return [self.action_space.sample() for _ in range(self.n_agents)]
        return action
