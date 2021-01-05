import os.path, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

for path in sys.path:
    print(path)

from supersuit import (
    resize_v0,
    color_reduction_v0,
    flatten_v0,
    normalize_obs_v0,
    dtype_v0,
)
from pettingzoo.butterfly import prison_v2
from ray.rllib.agents.qmix import QMixTrainer

# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper as GroupAgentsWrapper
from ray.tune import register_env
from ray import tune
import ray
import numpy as np
from gym.spaces import Dict, Discrete, Box, Tuple
from ray.rllib.agents.qmix.qmix_policy import ENV_STATE

# we can try to refactor the env, based on the custom wrapper in marlkit
from marlkit.envs.wrappers import ProxyEnv
from marlkit.envs.wrappers import MultiAgentEnv

env_wrapper = lambda x: flatten_v0(
    normalize_obs_v0(
        dtype_v0(
            resize_v0(color_reduction_v0(x), 4, 4),
            np.float32,
        )
    )
)


class MAEnv(MultiAgentEnv):
    def __init__(self):
        self.env = env_wrapper(prison_v2.parallel_env())
        self.base_observation_space = self.env.observation_spaces["prisoner_0"]
        self.n_agents = self.env.max_num_agents
        self.global_observation_spaces = self.get_global_observation_space(
            self.base_observation_space, self.n_agents
        )

    def get_global_observation_space(self, observation_space, n_agents):
        base_shape = list(observation_space.shape)
        new_shape = base_shape + [n_agents]
        new_shape = tuple(new_shape)
        high = np.stack([observation_space.high for _ in range(n_agents)], -1)
        low = np.stack([observation_space.low for _ in range(n_agents)], -1)
        return Box(low, high, shape=new_shape, dtype=observation_space.dtype)

    def reset(self, *args, **kwargs):
        obs_ = self.env.reset(*args, **kwargs)
        # global_state = self.get_global_state(obs_)
        return self._obs(obs_)

    def _obs(self, obs):
        obs_state = [(i, obs[f"prisoner_{i}"]) for i in range(self.env.max_num_agents)]
        obs_stack = np.stack([x[1] for x in obs_state], -1)
        obs_all = {}
        for idx, obs in obs_state:
            obs_all[idx] = {"obs": obs, ENV_STATE: obs_stack}
        return obs_all

    def get_global_state(self, obs):
        return np.stack(
            [obs[f"prisoner_{i}"] for i in range(self.env.max_num_agents)], -1
        )

    def step(self, act):
        act = {f"prisoner_{i}": act[i] for i in range(self.n_agents)}
        obs_, rewards, dones, _ = self.env.step(act)
        # global_state = self.get_global_state(obs_)
        obs = self._obs(obs_)
        dones = {"__all__": dones}
        rewards = {i: rewards[f"prisoner_{i}"] for i in range(self.n_agents)}
        return obs, rewards, dones, {}


env = env_wrapper(prison_v2.parallel_env())
maenv = MultiAgentEnv(env, rllib=True)

observation_spaces = Tuple(
    [maenv.multi_agent_observation_space for _ in range(maenv.max_num_agents)]
)
action_spaces = Tuple(
    [maenv.multi_agent_action_space for _ in range(maenv.max_num_agents)]
)

# clean up
ray.init()

register_env(
    "grouped_prison",
    lambda config: GroupAgentsWrapper(
        maenv,
        {"group_1": list(range(maenv.max_num_agents))},
        obs_space=observation_spaces,
        act_space=action_spaces,
    ),
)

mixer = "qmix"
time_s = 3000
local_dir = "rllibdata/"

config = {
    "env": "grouped_prison",
    "mixer": mixer,
    "rollout_fragment_length": 1,
}


results = tune.run(
    QMixTrainer,
    stop={
        "timesteps_total": time_s,
    },
    config=config,
    verbose=1,
    max_failures=1,
    local_dir=local_dir,
    checkpoint_at_end=True,
)
