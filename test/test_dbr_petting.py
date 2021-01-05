import os.path, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# from gym.envs.mujoco import HalfCheetahEnv
import gym

import marlkit.torch.pytorch_util as ptu
from marlkit.data_management.env_replay_buffer import MAEnvReplayBuffer
from marlkit.envs.wrappers import NormalizedBoxEnv
from marlkit.launchers.launcher_util import setup_logger
from marlkit.samplers.data_collector.marl_path_collector import MdpPathCollector
from marlkit.torch.sac.policies import MLPPolicy, MakeDeterministic, VAEPolicy
from marlkit.torch.sac.obtuse_discrete import OBTTrainer
from marlkit.torch.networks import FlattenMlp

# this is the marl variation!
from marlkit.torch.torch_marl_algorithm import TorchBatchMARLAlgorithm
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
    # rllib style envs
    expl_env = MultiAgentEnv(env_wrapper(prison_v2.parallel_env()))
    eval_env = MultiAgentEnv(env_wrapper(prison_v2.parallel_env()))

    obs_dim = expl_env.multi_agent_observation_space["obs"].low.size
    action_dim = expl_env.multi_agent_action_space.n

    M = variant["layer_size"]
    qf1 = FlattenMlp(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    policy = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    vae_pos_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        latent_dim=action_dim * 2,
    )
    vae_neg_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        latent_dim=action_dim * 2,
    )
    # be explicit with the mixer!
    eval_policy = MakeDeterministic(policy)

    # Mixer must always be None when in eval!
    eval_path_collector = MdpPathCollector(eval_env, eval_policy, mixer=None)
    expl_path_collector = MdpPathCollector(expl_env, policy, mixer=None)

    replay_buffer = MAEnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    trainer = OBTTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae_neg=vae_neg_policy,
        vae_pos=vae_pos_policy,
        **variant["trainer_kwargs"]
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


if __name__ == "__main__":
    # noinspection PyTypeChecker
    num_epochs = 50
    # num_epochs = 5

    variant = dict(
        algorithm="DBR",
        version="normal",
        layer_size=16,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=num_epochs,
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=100,
            max_path_length=100,
            batch_size=16,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger("test-dbr", variant=variant, base_log_dir="data")
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
