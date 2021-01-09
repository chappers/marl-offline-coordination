import numpy as np
from supersuit import (
    resize_v0,
    color_reduction_v0,
    flatten_v0,
    normalize_obs_v0,
    dtype_v0,
)
from pettingzoo.butterfly import prison_v2
from marlkit.envs.wrappers import MultiAgentEnv

env_wrapper = lambda x: flatten_v0(
    normalize_obs_v0(
        dtype_v0(
            resize_v0(color_reduction_v0(x), 4, 4),
            np.float32,
        )
    )
)

env = MultiAgentEnv(env_wrapper(prison_v2.parallel_env()), obs_agent_id=False, obs_last_action=False)
base_obs = env.reset()[0]["obs"]
base_state = env.reset()[0]["obs"]

env = MultiAgentEnv(env_wrapper(prison_v2.parallel_env()), global_pool=False)
full_obs = env.reset()[0]["obs"]
full_state = env.reset()[0]["state"]

assert full_obs.shape > base_obs.shape
assert full_state.shape > base_state.shape
