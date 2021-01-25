"""
A wrapper around lb-foraging for use in petting zoo style environments
"""

from env.lbforaging import foraging

import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np


class ForageEnv(gym.Env):
    def __init__(self, config={}, max_agents=None):
        self.env = foraging.ForagingEnv(**config)
        self.max_agents = len(self.env.players) if max_agents is None else max_agents
        self.action_space = [Discrete(len(foraging.environment.Action)) for _ in range(len(self.env.players))]
        self.observation_spaces = [self.env._get_observation_space() for _ in range(len(self.env.players))]
        self.possible_agents = list(range(len(self.env.players)))
        self.agents = list(range(len(self.env.players)))

    def gen_obs(self, n_obs):
        obs = {k: n_obs[k] for k in list(range(len(self.env.players)))}
        return obs

    def reset(self):
        n_obs = self.env.reset()
        return self.gen_obs(n_obs)

    def step(self, *args):
        n_obs, n_reward, n_done, info = self.env.step(*args)
        obs = self.gen_obs(n_obs)
        rew = {k: n_reward[k] for k in range(self.env.n_agents)}
        done = {"__all__": n_done[0]}
        return obs, rew, done, info


# lbf 8x8, 3p 1f full sight
# we increase the max episode steps?
n_agents = 3
s = 8
food = 1
base_config = dict(
    players=n_agents,
    max_player_level=3,
    field_size=(s, s),
    max_food=food,
    sight=8,
    max_episode_steps=100,
    force_coop=False,
)
