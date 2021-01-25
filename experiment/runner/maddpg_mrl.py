"""
An implementation of the independent actor critic style algorithm.

This one does uses GRU style actors but MLP critic (like COMA, and QMIX)
"""

import os.path, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import argparse

# from gym.envs.mujoco import HalfCheetahEnv
import gym

import marlkit.torch.pytorch_util as ptu
from marlkit.envs.wrappers import NormalizedBoxEnv
from marlkit.launchers.launcher_util import setup_logger
from marlkit.torch.sac.policies import MLPPolicy, MakeDeterministic
from marlkit.torch.networks import FlattenMlp
from marlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy

# MA DDPG
from marlkit.torch.ddpg.ma_ddpg_discrete import DDPGTrainer
from marlkit.policies.argmax import ArgmaxDiscretePolicy, Discretify
from marlkit.torch.networks import FlattenMlp, TanhMlpPolicy, TanhDiscreteMlpPolicy

# use the MARL versions!
from marlkit.torch.torch_marl_algorithm import TorchBatchMARLAlgorithm
from marlkit.exploration_strategies.epsilon_greedy import MAEpsilonGreedy
from marlkit.samplers.data_collector.marl_path_collector import MdpPathCollector
from marlkit.data_management.env_replay_buffer import FullMAEnvReplayBuffer

import numpy as np
from experiment.env import ENV_LOOKUP


def experiment(variant, train="pursuit", test="pursuit"):
    expl_env = ENV_LOOKUP[train]
    eval_env = ENV_LOOKUP[test]

    obs_dim = expl_env.multi_agent_observation_space["obs"].low.size
    action_dim = expl_env.multi_agent_action_space.n
    n_agents = expl_env.max_num_agents
    state_dim = eval_env.global_observation_space.low.size

    M = variant["layer_size"]
    N = variant["layer_mixer_size"]
    # N = variant["layer_size"]
    qf = FlattenMlp(
        input_size=obs_dim + action_dim + state_dim + action_dim * n_agents,
        output_size=action_dim,
        hidden_sizes=[N, N, N],
    )
    target_qf = FlattenMlp(
        input_size=obs_dim + action_dim + state_dim + action_dim * n_agents,
        output_size=action_dim,
        hidden_sizes=[N, N, N],
    )
    base_policy = TanhMlpPolicy(input_size=obs_dim, output_size=action_dim, hidden_sizes=[M, M, M])
    target_policy = TanhMlpPolicy(input_size=obs_dim, output_size=action_dim, hidden_sizes=[M, M, M])
    eval_policy = Discretify(base_policy, hard=True)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        MAEpsilonGreedy(expl_env.multi_agent_action_space, n_agents),
        Discretify(base_policy, hard=False),
    )

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
        qf=qf,
        target_qf=target_qf,
        policy=base_policy,
        target_policy=target_policy,
        use_joint_space=True,
        mrl=True,
        **variant["trainer_kwargs"],
    )
    algorithm = TorchBatchMARLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"],
    )
    algorithm.to(ptu.device)
    algorithm.train()


def run(train, test):
    base_agent_size = 64
    mixer_size = 32
    num_epochs = 1000
    buffer_size = 32
    max_path_length = 500
    eval_discard_incomplete = False if test in ["kaz"] else True

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=base_agent_size,
        layer_mixer_size=mixer_size,
        replay_buffer_size=buffer_size,
        algorithm_kwargs=dict(
            num_epochs=num_epochs,
            num_eval_steps_per_epoch=max_path_length * 5,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=max_path_length * 5,
            min_num_steps_before_training=1000,
            max_path_length=max_path_length,
            batch_size=32,  # this is number of episodes - not samples!
            eval_discard_incomplete=eval_discard_incomplete,
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
    )
    if test is None:
        test = train
    setup_logger(f"{train}-{test}-maddpgmrl", variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, train, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runner")
    parser.add_argument("--train", type=str, default="pursuit")
    parser.add_argument("--test", type=str, default="pursuit")
    args = parser.parse_args()
    train = args.train
    test = args.test
    run(train, test)
