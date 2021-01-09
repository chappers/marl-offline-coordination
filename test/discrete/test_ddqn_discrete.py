"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import os.path, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

for path in sys.path:
    print(path)
import copy

import gym

from marlkit.data_management.env_replay_buffer import EnvReplayBuffer
from marlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.policies.argmax import ArgmaxDiscretePolicy, Discretify
from marlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from marlkit.exploration_strategies.ou_strategy import OUStrategy
from marlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from marlkit.launchers.launcher_util import setup_logger
from marlkit.samplers.data_collector import MdpPathCollector
from marlkit.torch.networks import FlattenMlp, TanhMlpPolicy, TanhDiscreteMlpPolicy
from marlkit.torch.ddpg.ddpg import DDPGTrainer
import marlkit.torch.pytorch_util as ptu
from marlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(variant):
    expl_env = gym.make("CartPole-v0")
    eval_env = gym.make("CartPole-v0")

    # obs_dim = eval_env.observation_space.low.size
    # action_dim = eval_env.action_space.low.size
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, **variant["qf_kwargs"])
    base_policy = TanhMlpPolicy(input_size=obs_dim, output_size=action_dim, **variant["policy_kwargs"])
    expl_policy = Discretify(base_policy, hard=False)
    eval_policy = Discretify(base_policy, hard=True)
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(base_policy)
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        expl_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    replay_buffer = EnvReplayBuffer(variant["replay_buffer_size"], expl_env)
    trainer = DDPGTrainer(
        qf=qf, target_qf=target_qf, policy=base_policy, target_policy=target_policy, **variant["trainer_kwargs"]
    )
    algorithm = TorchBatchRLAlgorithm(
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
    variant = dict(
        algorithm_kwargs=dict(
            num_epochs=10,
            num_eval_steps_per_epoch=100,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=1000,
            max_path_length=100,
            batch_size=32,
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=1e-2,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        replay_buffer_size=int(1e6),
    )
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    setup_logger("test-ddqn-discrete", variant=variant)
    experiment(variant)
