import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np

from tqdm import tqdm

import argparse
from env.prison_simple.prison_simple import PrisonSimple


class PrisonEnv(gym.Env):
    def __init__(self):
        self.env = PrisonSimple()
        self.max_agents = self.env.n_agents
        self.action_space = [Discrete(3) for _ in range(self.max_agents)]
        self.observation_spaces = [Box(-1000, 1000, shape=(32,)) for _ in range(self.max_agents)]
        self.possible_agents = self.env.possible_agents
        self.agents = self.env.agents

    def reset(self):
        return self.env.reset()

    def step(self, *args):
        return self.env.step(*args)
