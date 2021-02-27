import logging

from collections import defaultdict, OrderedDict
import gym
from gym import spaces

from enum import Enum
import numpy as np

from typing import List, Tuple, Optional, Dict


class PrisonSimple(gym.Env):
    def __init__(self, n_agents=8, max_cycle=500):
        self.n_agents = n_agents
        self.agents = list(range(n_agents))
        self.possible_agents = self.agents
        self.agent_info = []
        self.n_sample = 30
        self.min_val = 0
        self.max_val = 30
        self.max_cycle = max_cycle
        self.num_steps = 0
        self.reset()

    def generate_agent(self):
        return {"loc": np.random.uniform(10, 20, 1), "left": False, "right": False, "first_touch": True}

    def reset(self):
        self.agent_info = {ag: self.generate_agent() for ag in self.agents}
        obs = self.gen_obs()
        self.num_steps = 0
        return obs

    def gen_agent_obs(self, info):
        obs = np.zeros(self.max_val + 2)
        max_obs_index = self.max_val - 1  # off by one index
        if info["left"]:
            obs[-2] += np.abs(np.random.normal(5, 2))
        if info["right"]:
            obs[-1] += np.abs(np.random.normal(5, 2))
        blob = np.random.normal(info["loc"], 2, self.n_sample).astype(int)
        for v in blob:
            indx = min(max(v, 0), max_obs_index)
            obs[indx] += np.abs(np.random.normal())
        return obs

    def gen_obs(self):
        return {ag: self.gen_agent_obs(self.agent_info[ag]) for ag in self.agents}

    def gen_rewards(self):
        agent_info = {}
        agent_reward = {}
        reward = 0
        for ag in self.agents:
            # print(self.agent_info[ag])
            if self.agent_info[ag]["left"] and self.agent_info[ag]["right"]:
                agent_reward[ag] = 2
                agent_info[ag] = self.generate_agent()
            elif (self.agent_info[ag]["left"] or self.agent_info[ag]["right"]) and self.agent_info[ag]["first_touch"]:
                agent_reward[ag] = 1
                agent_info[ag] = self.agent_info[ag]
                agent_info[ag]["first_touch"] = False  # only triggers once
            else:
                agent_reward[ag] = 0
                agent_info[ag] = self.agent_info[ag]
        self.agent_info = agent_info
        return agent_reward

    def step(self, action):
        # 0, nothing, 1, left, 2, right
        for ag in action.keys():
            act = action[ag]
            if act == 1:
                self.agent_info[ag]["loc"] -= np.abs(np.random.normal(5, 2))
            if act == 2:
                self.agent_info[ag]["loc"] += np.abs(np.random.normal(5, 2))
            self.agent_info[ag]["loc"] = min(max(self.agent_info[ag]["loc"], 0), self.max_val)
            if self.agent_info[ag]["loc"] < 2:
                self.agent_info[ag]["left"] = True
            if self.agent_info[ag]["loc"] > self.max_val - 2:
                self.agent_info[ag]["right"] = True

        # calculate reward
        reward = self.gen_rewards()  # this also resets successful agents (overloaded - to do refactor)...
        obs = self.gen_obs()
        self.num_steps += 1
        if self.num_steps == self.max_cycle:
            done = {"__all__": 1}
        else:
            done = {"__all__": 0}
        return obs, reward, done, {}


"""
env = PrisonSimple()
env.reset()
action = {'agent0': 1,
 'agent1': 1,
 'agent2': 1,
 'agent3': 1,
 'agent4': 1,
 'agent5': 1,
 'agent6': 1,
 'agent7': 1}
 action2 = {'agent0': 2,
 'agent1': 2,
 'agent2': 2,
 'agent3': 2,
 'agent4': 2,
 'agent5': 2,
 'agent6': 2,
 'agent7': 2}
 """
