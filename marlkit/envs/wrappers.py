import numpy as np
import itertools
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import Dict

from collections import deque


ENV_OBS = "obs"
ENV_STATE = "state"
ENV_STATE_0 = "state_0"
ENV_AGENT = "agent"  # binary vector representing if the agent was active or not


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        if hasattr(wrapped_env, "action_space"):
            self.action_space = self._wrapped_env.action_space
        elif hasattr(wrapped_env, "action_spaces"):
            self.action_space = self._wrapped_env.action_spaces
        if hasattr(wrapped_env, "observation_space"):
            self.observation_space = self._wrapped_env.observation_space
        elif hasattr(wrapped_env, "observation_spaces"):
            self.observation_space = self._wrapped_env.observation_spaces

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_env)


class HistoryEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, history_len):
        super().__init__(wrapped_env)
        self.history_len = history_len

        high = np.inf * np.ones(self.history_len * self.observation_space.low.size)
        low = -high
        self.observation_space = Box(
            low=low,
            high=high,
        )
        self.history = deque(maxlen=self.history_len)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history, reward, done, info

    def reset(self, **kwargs):
        state = super().reset()
        self.history = deque(maxlen=self.history_len)
        self.history.append(state)
        flattened_history = self._get_history().flatten()
        return flattened_history

    def _get_history(self):
        observations = list(self.history)

        obs_count = len(observations)
        for _ in range(self.history_len - obs_count):
            dummy = np.zeros(self._wrapped_env.observation_space.low.size)
            observations.append(dummy)
        return np.c_[observations]


class DiscretizeEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, num_bins):
        super().__init__(wrapped_env)
        low = self.wrapped_env.action_space.low
        high = self.wrapped_env.action_space.high
        action_ranges = [np.linspace(low[i], high[i], num_bins) for i in range(len(low))]
        self.idx_to_continuous_action = [np.array(x) for x in itertools.product(*action_ranges)]
        self.action_space = Discrete(len(self.idx_to_continuous_action))

    def step(self, action):
        continuous_action = self.idx_to_continuous_action[action]
        return super().step(continuous_action)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
        self,
        env,
        reward_scale=1.0,
        obs_mean=None,
        obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To " "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


class MultiAgentEnv(ProxyEnv):
    """
    This changes the dict items in parallel environments
    and changes the inputs to Dict observation spaces with
    global env states instead of the current one, and supports MAVEN
    style environments

    Petting zoo makes use of "possible_agents" as a key for max agents
    that will be present in the environment. It will make sure that things
    are appropriately ordered as well.

    As per all rllib style environments, we expect the inputs to be flattened?
    Now the output of the obs space will be
    [
        {'obs':<>, 'state':<>, 'state_0': <>}
    ]
    where it is ordered according the possible_agents

    rather than:
    {
        <agent>:<obs>
    }
    """

    def __init__(
        self,
        env,
        global_pool=False,
        stack=False,
        rllib=False,
        obs_agent_id=True,
        obs_last_action=True,
        max_num_agents=None,
    ):
        """
        - global pool enforces global state to be pooled obs size otherwise it will be stacked up to "max agents"
        - obs_agent_id is used in pymarl to allow parameter sharing and for agent to recognise each agent - onehot
        - obs_last_action is used in pymarl to encode the last action - onehot
        """
        ProxyEnv.__init__(self, env)
        self.max_num_agents = len(env.possible_agents) if max_num_agents is None else max_num_agents
        self.possible_agents = env.possible_agents
        self.global_pool = global_pool
        self.rllib = rllib
        self.obs_agent_id = obs_agent_id
        self.obs_last_action = obs_last_action
        self.agents = None
        self.stack = stack

        self.multi_agent_action_space = self.action_space[env.possible_agents[0]]
        # we need to expand the observation_space!
        self.base_observation_spaces = env.observation_spaces[env.possible_agents[0]]

        if self.obs_agent_id or self.obs_last_action:
            obs_space_shape = self.base_observation_spaces.low.shape
            assert len(obs_space_shape) == 1
            low = np.min(self.base_observation_spaces.low)
            high = np.max(self.base_observation_spaces.high)
            new_obs_shape = obs_space_shape[0]
            if self.obs_agent_id:
                # this allows for parameter sharing! so we don't need separate networks for
                # each agent.
                new_obs_shape += self.max_num_agents
            if self.obs_last_action:
                new_obs_shape += self.multi_agent_action_space.n
            self.observation_spaces = Box(low=low, high=high, shape=(new_obs_shape,))
        else:
            self.observation_spaces = env.observation_spaces[env.possible_agents[0]]

        # note that proxy env sets self.observation_space, not self.observation_spaces

        if self.global_pool:
            self.global_observation_space = self.observation_spaces
        elif self.stack:
            # print(self.observation_spaces.low)
            # print(self.observation_spaces.low.shape)
            low = np.concatenate([self.observation_spaces.low for _ in range(self.max_num_agents)], 0)
            high = np.concatenate([self.observation_spaces.high for _ in range(self.max_num_agents)], 0)
            self.global_observation_space = Box(low=low, high=high)
        else:
            low = np.stack([self.observation_spaces.low for _ in range(self.max_num_agents)], -1)
            high = np.stack([self.observation_spaces.high for _ in range(self.max_num_agents)], -1)
            low = low.flatten()
            high = low.flatten()
            self.global_observation_space = Box(low=low, high=high)

        self.initial_global_state = None
        self.previous_action = None
        self.current_action = None
        # define the default obs space for when an agent is nil?
        # its okay if there are nans, we'll handle later
        self.default_state = (self.observation_spaces.high + self.observation_spaces.low) / 2
        self.multi_agent_observation_space = Dict(
            {
                ENV_OBS: self.observation_spaces,
                ENV_STATE: self.global_observation_space,
                ENV_STATE_0: self.global_observation_space,
            }
        )

    def multi_obs(self, obs, reset=False):
        if reset:
            self.previous_action = None
            self.current_action = None
        obs_all = []
        active_agent = np.zeros(self.max_num_agents)
        for idx, agent in enumerate(self.possible_agents):
            obs_builder = obs.get(agent, self.default_state)
            if self.obs_last_action:
                action_vec = np.zeros(self.multi_agent_action_space.n)
                if self.previous_action is not None:
                    action_vec[self.previous_action[idx]] = 1
                obs_builder = np.hstack([obs_builder, action_vec])
                self.previous_action = self.current_action
            if self.obs_agent_id:
                agent_idx = np.zeros(self.max_num_agents)
                agent_idx[idx] = 1
                obs_builder = np.hstack([obs_builder, agent_idx])
            obs_all.append(obs_builder)
            if agent in self._wrapped_env.agents:
                active_agent[idx] = 1

        if self.global_pool:
            state = np.nanmean(obs_all, axis=0)
        elif self.stack:
            state = np.concatenate(obs_all, axis=0)
            state = np.nan_to_num(state)
        else:
            # don't flatten, leave it for the implementation
            obs_all = [x[: self.observation_spaces.low.shape[0]] for x in obs_all]
            state = np.stack(obs_all, axis=-1)
            # if there are nans, replace with 0
            state = np.nan_to_num(state)
            state = state.flatten()

        if reset:
            self.initial_global_state = state

        if self.rllib:
            next_obs = {}
            for idx, _ in enumerate(self.possible_agents):
                next_obs[idx] = {
                    ENV_OBS: obs_all[idx],
                    ENV_STATE: state,
                    ENV_STATE_0: self.initial_global_state,
                }

            return next_obs
        else:
            next_obs = []
            for idx, agent in enumerate(self.possible_agents):
                next_obs.append(
                    {
                        ENV_OBS: obs_all[idx],
                        ENV_STATE: state,
                        ENV_STATE_0: self.initial_global_state,
                        ENV_AGENT: active_agent,
                    }
                )
            return next_obs

    def multi_rewards(self, rewards):
        import warnings

        if self.rllib:
            rewards = {idx: rewards.get(ag, 0) for idx, ag in enumerate(self._wrapped_env.possible_agents)}
        else:
            rewards = [rewards.get(ag, 0) for idx, ag in enumerate(self._wrapped_env.possible_agents)]  # remove this!
        return rewards

    def multi_done(self, done):
        done = [done.get(ag, 0) for idx, ag in enumerate(self._wrapped_env.possible_agents)]
        if self.rllib:
            done = {"__all__": all(done)}
        return done

    def multi_info(self, info):
        if self.rllib:
            return {}
        else:
            try:
                info = [info.get(ag) for ag in self._wrapped_env.possible_agents]
                return info
            except:
                return {}

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        obs = self.multi_obs(obs, reset=True)
        return obs

    def step(self, action):
        self.current_action = action
        new_action = {}
        for idx, ag in enumerate(self._wrapped_env.possible_agents):
            new_action[ag] = action[idx]
        wrapped_step = self._wrapped_env.step(new_action)
        next_obs, reward, done, info = wrapped_step
        next_obs = self.multi_obs(next_obs)
        reward = self.multi_rewards(reward)
        done = self.multi_done(done)
        # deal with info later...
        info = self.multi_info(info)
        self.agents = self._wrapped_env.agents
        return next_obs, reward, done, info


class MultiEnv(MultiAgentEnv):
    def __init__(
        self,
        env_list,
        global_pool=False,
        stack=False,
        rllib=False,
        obs_agent_id=True,
        obs_last_action=True,
        max_num_agents=None,
    ):
        MultiAgentEnv.__init__(
            self,
            env_list[0],
            global_pool,
            stack,
            rllib,
            obs_agent_id,
            obs_last_action,
            max_num_agents,
        )

        self.env_list = []
        for e in env_list:
            self.env_list.append(
                MultiAgentEnv(e, global_pool, stack, rllib, obs_agent_id, obs_last_action, max_num_agents)
            )
        self.num_env = len(self.env_list)

    def reset(self):
        import random

        indx = random.choice(range(self.num_env))
        self.env_ = self.env_list[indx]
        return self.env_.reset()

    def step(self, *args, **kwargs):
        wrapped_step = self.env_.step(*args, **kwargs)
        next_obs, reward, done, info = wrapped_step
        self.agents = self.env_._wrapped_env.agents
        return next_obs, reward, done, info
