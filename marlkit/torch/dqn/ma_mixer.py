"""
An implementation of IQL just to familiarise myself with using this custom setup
"""

import numpy as np
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer


class DoubleDQNTrainer(DQNTrainer):
    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        state = batch["states"]
        active_agent = batch["active_agents"]
        # state_0 = batch["states_0"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # no need to worry about groups of games. in the IQL setting.
        obs = torch.from_numpy(np.stack(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.stack(terminals, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()
        rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
        active_agent = torch.from_numpy(np.stack(active_agent, axis=0)).float()

        """
        Compute loss
        """
        # rewards = rewards.reshape(-1, 1).float()
        # terminals = terminals.reshape(-1, 1).float()

        best_action_idxs = self.qf(next_obs).max(-1, keepdim=True)[1]
        # print(best_action_idxs.shape)
        # print(self.target_qf(next_obs).shape)
        target_q_values = self.target_qf(next_obs).gather(-1, best_action_idxs).detach()
        target_q_values = target_q_values.permute(0, 1, 3, 2)
        # print(target_q_values.shape)
        y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=-1, keepdim=True)

        state = torch.from_numpy(np.stack(state, 0)).float()

        if self.mixer is not None:
            # inputs needs to include batch['state']
            y_pred = y_pred.permute(0, 1, 3, 2)  # needs to match y_pred size
            # we need to pad out y_pred with agent active?
            if y_pred.shape != active_agent.shape:
                # need to concate along 0 axis...
                pad_y_pred_shape = list(active_agent.shape)
                pad_y_pred_shape[-1] = active_agent.shape[-1] - y_pred.shape[-1]
                y_pred = torch.cat([y_pred, torch.zeros(*pad_y_pred_shape)], axis=-1)
                y_target = torch.cat(
                    [y_target, torch.zeros(*pad_y_pred_shape)], axis=-1
                )

            y_pred = self.mixer(y_pred, state)
            y_target = self.target_mixer(y_target, state).detach()

        qf_loss = self.qf_criterion(y_pred, y_target)

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
            if self.mixer is not None:
                ptu.soft_update_from_to(
                    self.mixer, self.target_mixer, self.soft_target_tau
                )

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
