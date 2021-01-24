import logging

from collections import defaultdict, OrderedDict
import gym
from gym import spaces

from env.robotic_warehouse2.utils import MultiAgentActionSpace, MultiAgentObservationSpace

from enum import Enum
import numpy as np

from typing import List, Tuple, Optional, Dict

from env.robotic_warehouse2.warehouse import (
    Warehouse,
    _LAYER_AGENTS,
    _LAYER_SHELFS,
    Action,
    Direction,
)
import math


class GuideAction(Enum):
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    TOGGLE_LOAD = 5


class GuidedWarehouse(Warehouse):
    def __init__(self, coop_mode=False, guided=True, **kwargs):
        super().__init__(**kwargs)
        self.coop_mode = coop_mode
        self.guided = True  # this is on purpose because I haven't done the masks

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def _highway_lookup(self):
        self.shelf_location = (self.grid[_LAYER_SHELFS, :, :] > 0).copy()

    def reset(self):
        obs = super().reset()
        self._highway_lookup()
        return obs

    def coop_step(self, actions):
        """
        If agent is carrying, it can only move if another agent is next to it.
        Basically, override action to be NOOP in this case.
        """
        coop_action = []
        for agent, action in zip(self.agents, actions):
            if self.guided:
                if agent.carrying_shelf and GuideAction(action).name in [
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                ]:
                    if (
                        np.sum(
                            self.grid[
                                0, agent.y - 1 : agent.y + 2, agent.x - 1 : agent.x + 2
                            ]
                            > 0
                        )
                        <= 1
                    ):
                        action = GuideAction["NOOP"].value
            else:
                if agent.carrying_shelf and Action(action).name == "FORWARD":
                    if (
                        np.sum(
                            self.grid[
                                0, agent.y - 1 : agent.y + 2, agent.x - 1 : agent.x + 2
                            ]
                            > 0
                        )
                        <= 1
                    ):
                        action = Action["NOOP"].value
            coop_action.append(action)
        return coop_action

    def step(self, actions):
        """
        override to use guided actions
        """
        if self.coop_mode:
            actions = self.coop_step(actions)

        if self.guided:
            guide_action = []
            for agent, action in zip(self.agents, actions):
                if GuideAction(action).name in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    agent.dir = Direction[GuideAction(action).name]
                    action = Action["FORWARD"]
                elif GuideAction(action).name == "NOOP":
                    action = Action["NOOP"].value
                elif GuideAction(action).name == "TOGGLE_LOAD":
                    action = Action["TOGGLE_LOAD"].value
                guide_action.append(action)
            return super().step(guide_action)
        else:
            return super().step(actions)

    def _make_action_mask(self, agent):
        # creates a simple action mask.
        # agent = self.agents[agent_id - 1]
        action_mask = [1 for _ in range(len(GuideAction))]

        # simple version...could be more prescriptive, but agent needs to learn when its "on top" of
        # an appropriate shelf

        # if is carry, check whether up-down-left-right is a highway and mask out
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            agent.dir = Direction[GuideAction[direction].name]
            agent.req_action = Action.FORWARD
            target = agent.req_location(self.grid_size)
            if (
                not agent.carrying_shelf
                and target[0] == agent.x
                and target[1] == agent.y
            ):
                t1 = GuideAction[direction].value
                action_mask[t1] = 0
            elif (
                agent.carrying_shelf
                and not self._is_highway(target[0], target[1])
                and self.grid[_LAYER_SHELFS, target[1], target[0]] > 0
            ):
                t1 = GuideAction[direction].value
                action_mask[t1] = 0

        # shelf_in_queue = int(self.grid[_LAYER_SHELFS, agent.y, agent.x] in [x.id for x in self.request_queue])
        # action_mask[tl] = 1 - int(self._is_highway(agent.x, agent.y))

        shelf_on = 1 - int(self._is_highway(agent.x, agent.y))
        shelf_in_queue = int(
            self.grid[_LAYER_SHELFS, agent.y, agent.x]
            in [x.id for x in self.request_queue]
        )
        # shelf_any = int(self.grid[_LAYER_SHELFS, agent.y, agent.x] > 0)

        t1 = GuideAction["TOGGLE_LOAD"].value
        action_mask[t1] = 0

        # trigger conds where we want to load/unload
        if agent.carrying_shelf and shelf_on and not shelf_in_queue:
            action_mask[t1] = 1
        if not agent.carrying_shelf and shelf_on and shelf_in_queue:
            action_mask[t1] = 1
        # check toggle action: tl = GuideAction["TOGGLE_LOAD"].value

        return np.array(action_mask)

    def _make_obs(self, agent):
        """
        Make this more similar to microRTS and SMAC environments. This should make
        the observation vector "more dense" which should learn quicker

        TODO action masks; we can only use toggle load if shelf and agent are on same space + other conds
        """

        # min_x = agent.x - self.sensor_range
        # max_x = agent.x + self.sensor_range + 1

        # min_y = agent.y - self.sensor_range
        # max_y = agent.y + self.sensor_range + 1
        grid_max = max(self.grid_size)

        agents = self.grid[_LAYER_AGENTS, :, :]
        shelfs = self.grid[_LAYER_SHELFS, :, :]

        # get the closest self.n_agents
        # agent_mid_point = (agents.shape[0] // 2, agents.shape[1] // 2)
        agent_obs_size = min(self.n_agents, (self.sensor_range + 1) ** 2)
        agent_obs = np.zeros(
            5 * agent_obs_size
        )  # dist, x, y, is_carrying, is_on_top of shelf that is in the req. list
        agent_normalised = [
            (self.distance(y, x, agent.y, agent.x), y, x, agents[y, x],)
            for y, x in np.argwhere(agents > 0).tolist()
        ]
        agent_normalised.sort(key=lambda x: x[0])
        agent_normalised = [
            x for x in agent_normalised if x[0] <= self.sensor_range + 1
        ]
        for idx, a in enumerate(agent_normalised):
            a = list(a)
            # print(a)
            is_carrying = int(self.agents[a[3] - 1].carrying_shelf is not None)
            shelf_id = self.grid[
                _LAYER_SHELFS, self.agents[a[3] - 1].y, self.agents[a[3] - 1].x
            ]
            a[0] = (self.sensor_range + 1) - a[0] / (self.sensor_range + 1)
            a[1] = ((self.sensor_range + 1) - (a[1] - agent.y)) / (
                self.sensor_range + 1
            )
            a[2] = ((self.sensor_range + 1) - (a[2] - agent.x)) / (
                self.sensor_range + 1
            )
            a[3] = is_carrying
            a.append(int(shelf_id in [x.id for x in self.request_queue]))
            agent_obs[idx * 5 : idx * 5 + 5] = list(a)
            if idx > agent_obs_size:
                break

        # get the closest self.request_queue_size
        # shelf_base_obs = (2+self.request_queue_size)
        shelf_base_obs = 4
        shelf_obs_size = min(self.request_queue_size, (self.sensor_range + 1) ** 2)
        shelf_obs = np.zeros(
            shelf_base_obs * shelf_obs_size
        )  # dist, x, y, one_hot self.request_queue_size
        empty_shelf_obs = np.zeros(
            (shelf_base_obs - 1) * shelf_obs_size
        )  # dist, x, y, one_hot self.request_queue_size
        # shelf_mid_point = (shelfs.shape[0] // 2, shelfs.shape[1] // 2)
        # only get shelfs which are in the req. list
        shelf_normalised = [
            (self.distance(y, x, agent.y, agent.x), y, x, int(agents[y, x] > 0))
            for y, x in np.argwhere(
                np.isin(shelfs, [x.id for x in self.request_queue])
            ).tolist()
        ]
        shelf_normalised.sort(key=lambda x: x[0])
        shelf_normalised = [
            x for x in shelf_normalised if x[0] <= self.sensor_range + 1
        ]
        for idx, a in enumerate(shelf_normalised):
            a = list(a)
            # shelf_number = a[3]
            # shelf_ohe = [0 for _ in range(self.request_queue_size)]
            # print(shelf_number)
            # shelf_ohe[shelf_number] = 1
            a[0] = (self.sensor_range + 1) - a[0] / (self.sensor_range + 1)
            a[1] = ((self.sensor_range + 1) - (a[1] - agent.y)) / (
                self.sensor_range + 1
            )
            a[2] = ((self.sensor_range + 1) - (a[2] - agent.x)) / (
                self.sensor_range + 1
            )
            # a.extend(shelf_ohe)
            # print(a)
            # print(shelf_base_obs)
            shelf_obs[
                idx * shelf_base_obs : idx * shelf_base_obs + shelf_base_obs
            ] = list(a)
            if idx > shelf_obs_size:
                break

        # add empty shelves here as well, that are in highway...
        # ....

        shelf_on = 1 - int(self._is_highway(agent.x, agent.y))
        shelf_in_queue = int(
            self.grid[_LAYER_SHELFS, agent.y, agent.x]
            in [x.id for x in self.request_queue]
        )
        shelf_any = int(self.grid[_LAYER_SHELFS, agent.y, agent.x] > 0)
        shelf_return = not shelf_in_queue and shelf_any
        if shelf_return:
            empty_shelf_obs = empty_shelf_obs * 0
        else:
            empty_shelf_obs = empty_shelf_obs - 1

        if shelf_return:
            empty_shelf = (self.shelf_location > 0) & (
                self.grid[_LAYER_SHELFS, :, :] == 0
            )
            empty_shelf_normalised = [
                (self.distance(y, x, agent.y, agent.x), y, x)
                for y, x in np.argwhere(empty_shelf > 0).tolist()
            ]
            empty_shelf_normalised.sort(key=lambda x: x[0])
            empty_shelf_normalised = [
                x for x in empty_shelf_normalised if x[0] <= self.sensor_range + 1
            ]

            for idx, a in enumerate(empty_shelf_normalised):
                a = list(a)
                # shelf_number = a[3]
                # shelf_ohe = [0 for _ in range(self.request_queue_size)]
                # print(shelf_number)
                # shelf_ohe[shelf_number] = 1
                a[0] = (self.sensor_range + 1) - a[0] / (self.sensor_range + 1)
                a[1] = ((self.sensor_range + 1) - (a[1] - agent.y)) / (
                    self.sensor_range + 1
                )
                a[2] = ((self.sensor_range + 1) - (a[2] - agent.x)) / (
                    self.sensor_range + 1
                )
                # a.extend(shelf_ohe)
                # print(a)
                # print(shelf_base_obs)

                if idx * (shelf_base_obs - 1) > empty_shelf_obs.shape[0] - 1:
                    break

                empty_shelf_obs[
                    idx * (shelf_base_obs - 1) : idx * (shelf_base_obs - 1)
                    + (shelf_base_obs - 1)
                ] = list(a)
                if idx > shelf_obs_size:
                    break

        # add some information here on location of goal and if the current agent is carrying. Basically if
        # carry then goal direction and distance appears, otherwise it'll be -1 for distance parameter

        # ignore direction parameter, just move up down left right
        # mask out invalid load unload actions and ignore highways
        # int(self._is_highway(agent.x, agent.y)) is for unloading shelfs on highway
        is_carrying = int(agent.carrying_shelf is not None)

        # shelf is in req. queue

        info = [
            -1
            if not is_carrying
            else (
                grid_max
                - self.distance(agent.y, agent.x, self.goals[0][0], self.goals[0][1])
            )
            / (grid_max),
            ((grid_max) - (agent.y - self.goals[0][0])) / (grid_max),
            ((grid_max) - (agent.x - self.goals[0][1])) / (grid_max),
            -1
            if not is_carrying
            else (
                grid_max
                - self.distance(agent.y, agent.x, self.goals[1][0], self.goals[1][1])
            )
            / (grid_max),
            ((grid_max) - (agent.y - self.goals[1][0])) / (grid_max),
            ((grid_max) - (agent.x - self.goals[1][1])) / (grid_max),
            shelf_on,
            int(agent.carrying_shelf is not None),
            shelf_in_queue,
            shelf_any,
            shelf_return,
        ]

        action_mask = self._make_action_mask(agent)
        # if not self.guided:
        #     raise Exception("Not implemented for non guided")

        return {
            "obs": np.concatenate(
                [
                    agent_obs,
                    shelf_obs,
                    empty_shelf_obs,
                    np.array(info),
                    np.array(action_mask),
                ]
            ),
            "action_mask": action_mask,
        }
