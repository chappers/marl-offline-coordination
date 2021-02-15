"""
An implementation of IQL/Mixers using Quantile Regression 
This is for new paper.
"""

import numpy as np
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer
from marlkit.policies.argmax import MAArgmaxDiscretePolicy
import torch.optim as optim
from collections import OrderedDict


class CQLTrainer(DQNTrainer):
    num_quant = 200

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
        # to do support batch esp. as games are different lengths...
        """
        obs = torch.from_numpy(np.stack(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.stack(terminals, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()
        rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
        active_agent = torch.from_numpy(np.stack(active_agent, axis=0)).float()
        """

        def to_tensor(x, filter_n=None):
            try:
                if filter_n is None:
                    return torch.from_numpy(np.array(x, dtype=float)).float()
                else:
                    return torch.from_numpy(np.array(x[:filter_n], dtype=float)).float()
            except:
                x = [np.array(x_) for x_ in x]
                if filter_n is None:
                    x = np.stack(x, 0)
                else:
                    x = x[:filter_n]
                    x = np.stack(x, 0)
                return torch.from_numpy(x).float()

        total_qf_loss = []
        total_y_pred = []

        self.num_quant = self.qf.num_quant  # hack

        for b in range(len(obs)):
            try:
                rewards = to_tensor(batch["rewards"][b])
                terminals = to_tensor(batch["terminals"][b])
                obs = to_tensor(batch["observations"][b])
                state = to_tensor(batch["states"][b])
                active_agent = to_tensor(batch["active_agents"][b])
                # state_0 = batch["states_0"]
                actions = to_tensor(batch["actions"][b])
                next_obs = to_tensor(batch["next_observations"][b])
                next_states = to_tensor(batch["next_states"][b])
            except:
                filter_n = len(batch["observations"][b]) - 1
                rewards = to_tensor(batch["rewards"][b], filter_n)
                terminals = to_tensor(batch["terminals"][b], filter_n)
                obs = to_tensor(batch["observations"][b], filter_n)
                state = to_tensor(batch["states"][b], filter_n)
                active_agent = to_tensor(batch["active_agents"][b], filter_n)
                # state_0 = batch["states_0"]
                actions = to_tensor(batch["actions"][b], filter_n)
                next_obs = to_tensor(batch["next_observations"][b], filter_n)
                next_states = to_tensor(batch["next_states"][b], filter_n)

            """
            Compute loss
            """
            # rewards = rewards.reshape(-1, 1).float()
            # terminals = terminals.reshape(-1, 1).float()
            # print(next_obs.shape)
            action_next = self.qf(next_obs, return_action=True)
            # print(action_next.shape)
            best_action_idx = action_next.max(-1, keepdim=True)[1]
            # print(best_action_idx.shape)
            best_action_idx = torch.nn.functional.one_hot(best_action_idx, self.qf.action_size)
            # print(best_action_idx.shape)
            best_action_idx = best_action_idx.permute(0, 1, 3, 2)

            target_q_all = self.target_qf(next_obs).detach()
            target_q_values = target_q_all.gather(1, best_action_idx.repeat(1, 1, 1, self.num_quant))
            # target_q_values = target_q_values.permute(0, 2, 1)

            # print(target_q_values.shape, terminals.shape, rewards.shape)
            terminals = terminals.permute(0, 2, 1).unsqueeze(-1)
            rewards = rewards.permute(0, 2, 1).unsqueeze(-1)
            # print(target_q_values.shape, terminals.shape, rewards.shape)
            y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
            y_target_mixer = None
            # actions is a one-hot vector
            y_pred = self.qf(obs) * actions.unsqueeze(-1)
            # print(y_pred.shape)

            state = torch.from_numpy(np.stack(state, 0)).float()

            if self.mixer is None:
                y_target_mixer = None
                bellman_errors = y_target - y_pred  # for QR
                # eqn 9
                kappa = 1.0  # set this properly please
                huber_loss = torch.where(
                    bellman_errors.abs() < kappa,
                    0.5 * bellman_errors.pow(2),
                    kappa * (bellman_errors.abs() - 0.5 * kappa),
                )
                # quantile midpoints - lemma 2
                tau_hat = torch.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)
                qf_loss = huber_loss * ((tau_hat - bellman_errors.detach() < 0).float().abs())
                qf_loss = qf_loss.mean()
            else:
                # apply mixer to calculate qf loss.
                # inputs needs to include batch['state']

                y_pred = y_pred.mean(3).permute(0, 2, 1)
                y_target = y_target.mean(3).permute(0, 2, 1)
                # print(y_pred.shape, y_target.shape)

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

                # apply loss
                bellman_errors = y_target_mixer - y_pred  # for QR
                # eqn 9
                kappa = 1.0  # set this properly please
                huber_loss = torch.where(
                    bellman_errors.abs() < kappa,
                    0.5 * bellman_errors.pow(2),
                    kappa * (bellman_errors.abs() - 0.5 * kappa),
                )
                # quantile midpoints - lemma 2
                tau_hat = torch.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)
                qf_loss = huber_loss * ((tau_hat - bellman_errors.detach() < 0).float().abs())
                qf_loss = qf_loss.mean()

            """
            Update networks
            """
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            total_qf_loss.append(ptu.get_numpy(qf_loss))
            total_y_pred.append(ptu.get_numpy(y_pred))

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)
            if self.mixer is not None:
                ptu.soft_update_from_to(self.mixer, self.target_mixer, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        These are all wrong...maybe rework them in a bit.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(total_qf_loss)
            try:
                self.eval_statistics.update(
                    create_stats_ordered_dict(
                        "Y Predictions",
                        total_y_pred,
                    )
                )
            except:
                pass

        self._n_train_steps_total += 1
