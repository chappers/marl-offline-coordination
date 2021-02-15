from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.torch_rl_algorithm import MATorchTrainer
from marlkit.policies.argmax import MAArgmaxDiscretePolicy


class DQNTrainer(MATorchTrainer):
    def __init__(
        self,
        qf,
        target_qf,
        policy=None,
        mixer=None,
        target_mixer=None,
        learning_rate=1e-3,
        soft_target_tau=1e-3,
        target_update_period=1,
        qf_criterion=None,
        discount=0.99,
        reward_scale=1.0,
        # TODO
        use_shared_experience=False,
        n_agents=None,
        state_dim=None,
        action_dim=None,
        obs_dim=None,
        mrl=False,
        inverse_weight=False,
        num_quant=None,
    ):
        super().__init__()
        self.qf = qf
        self.num_quant = num_quant
        if policy is None:
            self.policy = MAArgmaxDiscretePolicy(self.qf)
        else:
            self.policy = policy

        self.mixer = mixer
        self.target_mixer = target_mixer
        self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        if use_shared_experience:
            assert mixer is None, "Shared experience only makes sense in IQL!"
        self.use_shared_experience = use_shared_experience
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.mrl = mrl
        self.inverse_weight = inverse_weight

    def train_from_torch(self, batch):
        rewards = batch["rewards"] * self.reward_scale
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Compute loss
        """

        target_q_values = self.target_qf(next_obs).detach().max(1, keepdim=True)[0]
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Y Predictions",
                    ptu.get_numpy(y_pred),
                )
            )
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        if self.mixer is not None:
            return [self.qf, self.target_qf, self.mixer, self.target_mixer]
        else:
            return [self.qf, self.target_qf]

    def get_snapshot(self):
        if self.mixer is not None:
            return dict(
                qf=self.qf,
                target_qf=self.target_qf,
                mixer=self.mixer,
                target_mixer=self.target_mixer,
            )
        else:
            return dict(
                qf=self.qf,
                target_qf=self.target_qf,
            )
