"""
pip install git+https://github.com/uoe-agents/robotic-warehouse.git
and check this one too: https://github.com/uoe-agents/lb-foraging
"""

# import robotic_warehouse
import env.robotic_warehouse
from env.robotic_warehouse import warehouse
from env.robotic_warehouse.warehouse import Warehouse
from env.robotic_warehouse.smac_warehouse import GuidedWarehouse, GuideAction
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np

from tqdm import tqdm

import argparse

n_agents = 4

# (1, 3) -> tiny
shelf_columns = 3
shelf_rows = 1


class RwareEnv(gym.Env):
    def __init__(self, config={}, max_agents=None):
        self.env = GuidedWarehouse(**config)
        self.max_agents = self.env.n_agents if max_agents is None else max_agents
        self.action_space = [Discrete(len(GuideAction)) for _ in range(self.env.n_agents)]
        self.observation_spaces = [
            Box(-np.inf, np.inf, shape=self.env.reset()[0]["obs"].shape) for _ in range(self.env.n_agents)
        ]
        self.possible_agents = list(range(self.env.n_agents))
        self.agents = list(range(self.env.n_agents))

    def gen_obs(self, n_obs):
        # obs = {
        #     k: {"obs": n_obs[k], "action_mask": self._make_action_mask(k)}
        #     for k in range(self.env.n_agents)
        # }

        obs = {k: n_obs[k]["obs"] for k in range(self.env.n_agents)}
        return obs

    def reset(self):
        n_obs = self.env.reset()
        # self.max_normalise = max(self.env.grid[1, :, :].shape)

        obs = self.gen_obs(n_obs)
        return obs

    def render(self):
        self.env.render()

    @staticmethod
    def pad_action(self, action, key):
        try:
            return action[key]
        except:
            return 0

    def step(self, actions):
        # actions = [self.pad_action(actions, k) for k in range(self.env.n_agents)]
        actions = [actions[k] for k in range(self.env.n_agents)]
        n_obs, n_reward, n_done, info = self.env.step(actions)
        # n_obs = []
        # for obs in n_obs_:
        #     obs[:2] /= self.max_normalise
        #     n_obs.append(obs)

        # global_obs = np.concatenate([self.env.grid[1, :, :].flatten()/self.max_normalise*self.max_normalise*2, np.concatenate(n_obs)])

        obs = self.gen_obs(n_obs)
        rew = {k: n_reward[k] for k in range(self.env.n_agents)}
        done = {"__all__": n_done[0]}
        return obs, rew, done, info

    def play(self, actions):
        # for human play only
        if type(actions) is int:
            actions = [actions]
        o, r, _, _ = self.step(actions)
        am = o[0]["action_mask"]
        am = {
            "up-1": am[1],
            "down-2": am[2],
            "left-3": am[3],
            "right-4": am[4],
            "load-5": am[5],
        }
        self.render()
        return o[0]["obs"], am, r


# change this or make it configurable
"""
env = RwareEnv()
"""

# tiny 4p
base_config = dict(
    shelf_columns=shelf_columns,
    column_height=8,
    shelf_rows=shelf_rows,
    n_agents=n_agents,
    msg_bits=0,
    sensor_range=2,
    request_queue_size=n_agents,
    max_inactivity_steps=None,
    max_steps=500,
    reward_type=1,
)
