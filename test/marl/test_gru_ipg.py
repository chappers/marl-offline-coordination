"""
An implementation of the independent actor critic style algorithm.

This one does uses GRU style actors but MLP critic (like COMA, and QMIX)
"""

import os.path, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


# from gym.envs.mujoco import HalfCheetahEnv
import gym
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.envs.wrappers import NormalizedBoxEnv
from marlkit.launchers.launcher_util import setup_logger
from marlkit.torch.sac.policies import MLPPolicy, MakeDeterministic
from marlkit.torch.networks import FlattenMlp
from marlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy

# MA DDPG
from marlkit.torch.ddpg.ma_gru_ddpg_discrete import DDPGTrainer
from rlkit.policies.argmax import ArgmaxDiscretePolicy, Discretify
from marlkit.torch.networks import FlattenMlp, TanhMlpPolicy, TanhDiscreteMlpPolicy
from marlkit.torch.networks import RNNNetwork
from marlkit.policies.recurrent import RecurrentPolicy

# use the MARL versions!
from marlkit.torch.torch_marl_algorithm import TorchBatchMARLAlgorithm
from marlkit.exploration_strategies.epsilon_greedy import MAEpsilonGreedy
from marlkit.samplers.data_collector.marl_path_collector import MdpPathCollector
from marlkit.data_management.env_replay_buffer import FullMAEnvReplayBuffer

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


def experiment(variant):
    expl_env = MultiAgentEnv(env_wrapper(prison_v2.parallel_env()))
    eval_env = MultiAgentEnv(env_wrapper(prison_v2.parallel_env()))

    n_agents = expl_env.max_num_agents

    obs_dim = expl_env.multi_agent_observation_space["obs"].low.size
    action_dim = expl_env.multi_agent_action_space.n

    M = variant["layer_size"]
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    base_policy = RNNNetwork(hidden_sizes=M, input_size=obs_dim, output_size=action_dim, output_activation=torch.tanh)
    # THIS NEEDS TO BE GUMBEL SOFTMAX

    eval_policy = RecurrentPolicy(base_policy, use_gumbel_softmax=True, eval_policy=True)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        MAEpsilonGreedy(expl_env.multi_agent_action_space, n_agents),
        RecurrentPolicy(base_policy, use_gumbel_softmax=True, eval_policy=False),
    )
    target_qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    target_policy = RNNNetwork(hidden_sizes=M, input_size=obs_dim, output_size=action_dim, output_activation=torch.tanh)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    replay_buffer = FullMAEnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    trainer = DDPGTrainer(
        qf=qf, target_qf=target_qf, policy=base_policy, target_policy=target_policy, **variant["trainer_kwargs"]
    )
    algorithm = TorchBatchMARLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()


def test():
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=32,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=10,
            num_eval_steps_per_epoch=10,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=10,
            min_num_steps_before_training=10,
            max_path_length=20,
            batch_size=32,
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
    )
    setup_logger("test-ddqn", variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)


if __name__ == "__main__":
    test()
