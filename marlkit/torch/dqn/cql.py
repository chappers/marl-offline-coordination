"""
cql for dqn is built over qr dqn.
"""

import numpy as np
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.dqn import DQNTrainer


class CQLTrainer(DQNTrainer):
    num_quant = 200
    min_q_weight = 1.0

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # qf in this instance goes something like
        # [B, num_actions, num_quantiles]

        """
        Compute loss
        """
        """
        best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
        target_q_values = self.target_qf(next_obs).gather(1, best_action_idxs).detach()
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)
        """
        self.num_quant = self.qf.num_quant  # hack

        action_next = self.qf(next_obs)
        best_action_idx = action_next.mean(2).max(1)[1]  # hope this is right
        best_action_idx = torch.nn.functional.one_hot(best_action_idx, self.qf.action_size)
        target_q_all = self.target_qf(next_obs).detach()
        # print(target_q_all.shape, best_action_idx.shape)
        target_q_values = target_q_all.gather(1, best_action_idx.unsqueeze(-1).repeat(1, 1, self.num_quant))
        # print(target_q_values.shape, terminals.shape, rewards.shape)
        y_target = rewards.unsqueeze(-1) + (1.0 - terminals.unsqueeze(-1)) * self.discount * target_q_values
        y_pred = self.qf(obs) * actions.unsqueeze(-1)  # actions are one hot

        bellman_errors = y_target - y_pred

        # eqn 9
        kappa = 1.0  # set this properly please
        huber_loss = torch.where(
            bellman_errors.abs() < kappa,
            0.5 * bellman_errors.pow(2),
            kappa * (bellman_errors.abs() - 0.5 * kappa),
        )

        # quantile midpoints - lemma 2
        tau_hat = torch.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)
        loss = huber_loss * ((tau_hat - bellman_errors.detach() < 0).float().abs())
        loss = loss.mean()

        # Add CQL Loss here see: https://github.com/aviralkumar2907/CQL/blob/master/atari/batch_rl/multi_head/quantile_agent.py
        # and also CQL pytorch variation for SAC
        replay_chosen_q = self.qf(obs) * actions.unsqueeze(-1)
        dataset_expec = replay_chosen_q.mean()
        negative_sampling = torch.logsumexp(replay_chosen_q, dim=1).mean()

        min_q_loss = (negative_sampling - dataset_expec) * self.min_q_weight

        qf_loss = loss + min_q_loss

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft target network updates
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
