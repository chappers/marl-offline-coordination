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
from pettingzoo.sisl import waterworld_v3
from pettingzoo.sisl import multiwalker_v6
from pettingzoo.butterfly import prison_v2
from pettingzoo.butterfly import knights_archers_zombies_v5
from pettingzoo.butterfly import pistonball_v3
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2
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

    kaz = MultiAgentEnv(grid_wrapper(knights_archers_zombies_v5.parallel_env()), global_pool=False),
    kaz_mix = MultiEnv(
        [
            grid_wrapper(knights_archers_zombies_v5.parallel_env(num_knights=1, num_archers=1)),
            grid_wrapper(knights_archers_zombies_v5.parallel_env(num_knights=1, num_archers=2)),
            grid_wrapper(knights_archers_zombies_v5.parallel_env(num_knights=2, num_archers=1)),
            grid_wrapper(knights_archers_zombies_v5.parallel_env(num_knights=2, num_archers=2)),
        ],
        max_num_agents=4,
        global_pool=False,
    ),

    pistonball = MultiAgentEnv(grid_wrapper(pistonball_v3.parallel_env()), global_pool=False),
    pistonball_mix = MultiEnv(
        [
            grid_wrapper(pistonball_v3.parallel_env(n_pistons=20)),
            grid_wrapper(pistonball_v3.parallel_env(n_pistons=15)),
            grid_wrapper(pistonball_v3.parallel_env(n_pistons=10)),
            grid_wrapper(pistonball_v3.parallel_env(n_pistons=5)),
        ],
        max_num_agents=20,
        global_pool=False,
    ),
    

    pursuit = MultiAgentEnv(simple_wrapper(pursuit_v3.parallel_env())),
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

    waterworld = MultiAgentEnv(simple_wrapper(waterworld_v3.parallel_env())),
    waterworld_multi = MultiEnv(
        [
            simple_wrapper(waterworld_v3.parallel_env(n_pursuers=2, n_evaders=2, n_poison=4)),
            simple_wrapper(waterworld_v3.parallel_env(n_pursuers=3, n_evaders=3, n_poison=6)),
            simple_wrapper(waterworld_v3.parallel_env(n_pursuers=4, n_evaders=4, n_poison=8)),
            simple_wrapper(waterworld_v3.parallel_env(n_pursuers=5, n_evaders=5, n_poison=10)),
        ],
        max_num_agents=5,
        global_pool=False,
    ),

    multiwalker = MultiAgentEnv(simple_wrapper(multiwalker_v6.parallel_env())),
    multiwalker_multi = MultiEnv(
        [
            simple_wrapper(multiwalker_v6.parallel_env(n_walkers=3)),
            simple_wrapper(multiwalker_v6.parallel_env(n_walkers=2)),
            simple_wrapper(multiwalker_v6.parallel_env(n_walkers=1)),
        ],
        max_num_agents=3,
        global_pool=False,
    ),

    spread = MultiAgentEnv(simple_wrapper(simple_spread_v2.parallel_env())),
    spread_multi = MultiEnv(
        [
            simple_wrapper(simple_spread_v2.parallel_env(N=3)),
            simple_wrapper(simple_spread_v2.parallel_env(N=2)),
            simple_wrapper(simple_spread_v2.parallel_env(N=1)),
        ],
        max_num_agents=3,
        global_pool=False,
    ),

    reference = MultiAgentEnv(simple_wrapper(simple_reference_v2.parallel_env())),
)