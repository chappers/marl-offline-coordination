"""
This assumes the similar multi-agent controller setup to pymarl and rllib. 
"""

import abc
import copy

# Visualization
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from marlkit.torch import pytorch_util as ptu

import gtimer as gt
from marlkit.core.rl_algorithm import BaseRLAlgorithm, BaseMARLAlgorithm
from marlkit.core.rl_algorithm import eval_util
from marlkit.data_management.replay_buffer import MAReplayBuffer
from marlkit.samplers.data_collector import PathCollector
from marlkit.samplers.data_collector.marl_path_collector import MdpPathCollector
import numpy as np
from marlkit.torch.core import np_to_pytorch_batch

import torch

# constants for the agents so that they can do things related to
# MAVEN and QMIX environments
# if unavailable (the default), construct and save manually here...
ENV_OBS = "obs"
ENV_STATE = "state"
ENV_STATE_0 = "state_0"

import warnings

warnings.warn("gt set to be nonunique!")
gt.set_def_unique(False)


def get_flat_params(model):
    params = []
    for param in model.parameters():
        # import ipdb; ipdb.set_trace()
        params.append(param.data.cpu().numpy().reshape(-1))
    return np.concatenate(params)


def set_flat_params(model, flat_params, trainable_only=True):
    idx = 0
    # import ipdb; ipdb.set_trace()
    for p in model.parameters():
        flat_shape = int(np.prod(list(p.data.shape)))
        flat_params_to_assign = flat_params[idx : idx + flat_shape]

        if len(p.data.shape):
            p.data = ptu.tensor(flat_params_to_assign.reshape(*p.data.shape))
        else:
            p.data = ptu.tensor(flat_params_to_assign[0])
        idx += flat_shape
    return model


class BatchMARLAlgorithm(BaseMARLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
        self,
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: PathCollector,
        evaluation_data_collector: PathCollector,
        replay_buffer: MAReplayBuffer,
        batch_size,
        max_path_length,
        num_epochs,
        num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop,
        num_trains_per_train_loop,
        num_train_loops_per_epoch=1,
        min_num_steps_before_training=0,
        q_learning_alg=False,
        eval_both=False,
        batch_rl=False,
        num_actions_sample=10,
        # this is needed for later - esp. for QMIX and MAVEN envs
        # where the mixing network can only accept fixed sizes
        flatten_global_state=False,
        eval_discard_incomplete=True,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.batch_rl = batch_rl
        self.q_learning_alg = q_learning_alg
        self.eval_both = eval_both
        self.num_actions_sample = num_actions_sample
        self.eval_discard_incomplete = eval_discard_incomplete

        ### Reserve path collector for evaluation, visualization
        # if hasattr(a, 'property'):
        self._reserve_path_collector = MdpPathCollector(
            env=evaluation_env,
            policy=self.trainer.policy,
            mixer=self.trainer.mixer if hasattr(self.trainer, "mixer") else None,
        )

        self.running_loss = None
        self.running_loss_count = 0
        self.running_loss_target = 10
        self.running_loss_min_epoch = 100

    def policy_fn(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            state = ptu.from_numpy(obs.reshape(1, -1)).repeat(self.num_actions_sample, 1)
            action, _, _, _, _, _, _, _ = self.trainer.policy(state)
            q1 = self.trainer.qf1(state, action)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten()

    def policy_fn_discrete(self, obs):
        with torch.no_grad():
            obs = ptu.from_numpy(obs.reshape(1, -1))
            q_vector = self.trainer.qf1.q_vector(obs)
            action = q_vector.max(1)[1]
        ones = np.eye(q_vector.shape[1])
        return ptu.get_numpy(action).flatten()

    def _train(self):
        if self.min_num_steps_before_training > 0 and not self.batch_rl:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            if self.q_learning_alg:
                policy_fn = self.policy_fn
                try:
                    if self.trainer.discrete:
                        policy_fn = self.policy_fn_discrete
                except:
                    pass

                # for MARL - and petting zoo, set discard_incomplete_paths to False
                # as most of the environments you die and does not terminate correctly?
                if (epoch % 5) == 0:
                    self.eval_data_collector.collect_new_paths(
                        policy_fn,
                        self.max_path_length,
                        self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=self.eval_discard_incomplete,
                    )
            else:
                if (epoch % 5) == 0:
                    self.eval_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=self.eval_discard_incomplete,
                    )
            gt.stamp("evaluation sampling")

            for _ in range(self.num_train_loops_per_epoch):
                if not self.batch_rl:
                    # Sample new paths only if not doing batch rl
                    new_expl_paths = self.expl_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_expl_steps_per_train_loop,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp("exploration sampling", unique=False)

                    self.replay_buffer.add_paths(new_expl_paths)
                    gt.stamp("data storing", unique=False)
                elif self.eval_both:
                    # Now evaluate the policy here:
                    policy_fn = self.policy_fn
                    if self.trainer.discrete:
                        policy_fn = self.policy_fn_discrete
                    new_expl_paths = self.expl_data_collector.collect_new_paths(
                        policy_fn,
                        self.max_path_length,
                        self.num_eval_steps_per_epoch,
                        discard_incomplete_paths=self.eval_discard_incomplete,
                    )

                    gt.stamp("policy fn evaluation")

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp("training", unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
