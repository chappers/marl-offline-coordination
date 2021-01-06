"""
The equivalent of the multi-agent controller in pymarl.
"""
from collections import deque, OrderedDict

from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.samplers.rollout_functions import (
    marl_rollout,
    multitask_rollout,
    function_rollout,
)
from marlkit.samplers.data_collector.base import PathCollector
import numpy as np


class MdpPathCollector(PathCollector):
    """
    The challenge with the path collector is that the number of agents
    for each path might be different. If its blank, either it can't be stored
    or it needs to be excluded at training time in the get.
    """

    def __init__(
        self,
        env,
        policy,
        mixer=None,
        max_num_epoch_paths_saved=None,
        render=False,
        sparse_reward=False,
        render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        # should NOT be used if in "eval" - only expl.
        self._mixer = None  # for qmix, maven and variations
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._sparse_reward = sparse_reward

    def update_policy(self, new_policy):
        self._policy = new_policy

    def collect_new_paths(
        self,
        max_path_length,
        num_steps,
        discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = marl_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            # print("path_actions", path["actions"])
            path_len = path["actions"].shape[0]
            if (
                path_len != max_path_length
                and not path["terminals"][-1]
                and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len

            ## Used to sparsify reward
            if self._sparse_reward:
                random_noise = np.random.normal(size=path["rewards"].shape)
                path["rewards"] = path["rewards"] + 1.0 * random_noise
                # bins = np.array([-10, -0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                # temp_rewards = np.cast(path['rewards']/2.0, )
                # temp_rewards = (path['rewards'] > 1.0)
                # path['rewards'] = temp_rewards.astype(np.float32)

            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path["actions"]) for path in self._epoch_paths]
        stats = OrderedDict(
            [
                ("num steps total", self._num_steps_total),
                ("num paths total", self._num_paths_total),
            ]
        )
        stats.update(
            create_stats_ordered_dict(
                "path length",
                path_lens,
                always_show_all_stats=True,
            )
        )
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )
