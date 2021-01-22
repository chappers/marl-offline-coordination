# Environment definitions

import numpy as np
from supersuit import (
    resize_v0,
    color_reduction_v0,
    flatten_v0,
    normalize_obs_v0,
    dtype_v0,
    pad_observations_v0,
    pad_action_space_v0,
)
from pettingzoo.sisl import pursuit_v3
from pettingzoo.butterfly import prison_v2
from marlkit.envs.wrappers import MultiAgentEnv, MultiEnv

resize_size = 32  # used int he petting zoo paper - and just to make it easy

simple_wrapper = lambda x: flatten_v0(
    normalize_obs_v0(
        dtype_v0(
            pad_observations_v0(
                pad_action_space_v0(
                    x,
                )
            ),
            np.float32,
        )
    )
)

grid_wrapper = lambda x: flatten_v0(
    normalize_obs_v0(
        dtype_v0(
            pad_observations_v0(
                pad_action_space_v0(
                    resize_v0(color_reduction_v0(x), resize_size, resize_size),
                )
            ),
            np.float32,
        )
    )
)

ENV_LOOKUP = dict(
    prison = MultiAgentEnv(grid_wrapper(prison_v2.parallel_env()), global_pool=False),
    prison_mix = MultiEnv(
        [
            grid_wrapper(prison_v2.parallel_env(n_floors=1)),
            grid_wrapper(prison_v2.parallel_env(n_floors=2)),
            grid_wrapper(prison_v2.parallel_env(n_floors=3)),
            grid_wrapper(prison_v2.parallel_env(n_floors=4)),
        ],
        max_num_agents=8,
        global_pool=False,
    ),

    pursuit = MultiAgentEnv(env_wrapper(pursuit_v3.parallel_env())),
    pursuit_multi = MultiEnv(
        [
            simple_wrapper(pursuit_v3.parallel_env(n_pursuers=2, n_evaders=8)),
            simple_wrapper(pursuit_v3.parallel_env(n_pursuers=4, n_evaders=15)),
            simple_wrapper(pursuit_v3.parallel_env(n_pursuers=6, n_evaders=22)),
            simple_wrapper(pursuit_v3.parallel_env(n_pursuers=8, n_evaders=30)),
        ],
        max_num_agents=8,
        global_pool=False,
    ),
    
)