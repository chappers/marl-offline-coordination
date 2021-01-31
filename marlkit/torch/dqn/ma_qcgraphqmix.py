"""
An implementation of IQL just to familiarise myself with using this custom setup
"""

import numpy as np
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer
from marlkit.policies.argmax import MAArgmaxDiscretePolicy
import torch.optim as optim
from collections import OrderedDict


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
        # to do support batch esp. as games are different lengths...
        """
        obs = torch.from_numpy(np.stack(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.stack(terminals, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()
        rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
        active_agent = torch.from_numpy(np.stack(active_agent, axis=0)).float()
        """

        def to_tensor(x):
            try:
                return torch.from_numpy(np.array(x, dtype=float)).float()
            except:
                x = np.stack([x_.flatten()[np.newaxis, :] for x_ in x], 0)
                return torch.from_numpy(x).float()

        total_qf_loss = []
        total_y_pred = []

        for b in range(len(obs)):
            rewards = to_tensor(batch["rewards"][b])
            terminals = to_tensor(batch["terminals"][b])
            obs = to_tensor(batch["observations"][b])
            state = to_tensor(batch["states"][b])
            active_agent = to_tensor(batch["active_agents"][b])
            # state_0 = batch["states_0"]
            actions = to_tensor(batch["actions"][b])
            next_obs = to_tensor(batch["next_observations"][b])

            """
            Compute loss
            """
            # rewards = rewards.reshape(-1, 1).float()
            # terminals = terminals.reshape(-1, 1).float()

            obs_qf, hidden_states = self.qf(next_obs, return_hidden=True)
            best_action_idxs = obs_qf.max(-1, keepdim=True)[1]
            # print(best_action_idxs.shape)
            # print(self.target_qf(next_obs).shape)
            next_obs_qf, target_hidden_states = self.target_qf(next_obs, return_hidden=True)
            target_q_values = next_obs_qf.gather(-1, best_action_idxs).detach()
            target_q_values = target_q_values.permute(0, 2, 1)
            # print(target_q_values.shape)
            y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
            y_target = y_target.detach()
            # actions is a one-hot vector
            obs_qs = self.qf(obs)
            y_pred = torch.sum(obs_qs * actions, dim=-1, keepdim=True)

            state = torch.from_numpy(np.stack(state, 0)).float()

            if self.mixer is not None:
                # inputs needs to include batch['state']
                y_pred = y_pred.permute(0, 2, 1)  # needs to match y_pred size
                # we need to pad out y_pred with agent active?
                if y_pred.shape != active_agent.shape:
                    # need to concate along 0 axis...
                    pad_y_pred_shape = list(active_agent.shape)
                    pad_y_pred_shape[-1] = active_agent.shape[-1] - y_pred.shape[-1]
                    y_pred = torch.cat([y_pred, torch.zeros(*pad_y_pred_shape)], axis=-1)
                    y_target = torch.cat([y_target, torch.zeros(*pad_y_pred_shape)], axis=-1)

                y_pred = self.mixer(y_pred, state, hidden_states)
                y_target = self.target_mixer(y_target, state, target_hidden_states).detach()

            if self.use_shared_experience:
                # assume lambda = 1 as per paper, so we only need to iterate and not do the top part
                n_agents = obs_qs.shape[-2]
                # policy_loss_ = (action_probs * (alpha * log_pis - q_new_actions))
                y_target = y_target.permute(0, 2, 1)
                qf_loss_ = (y_pred - y_target) ** 2

                pis = torch.softmax(obs_qs.detach(), -1)
                log_pi = torch.log(pis)

                qf_loss = None

                for ag in range(n_agents):
                    # iterate through all of them...
                    if qf_loss is None:
                        qf_loss = (
                            torch.exp(torch.exp(log_pi - log_pi[:, :, [ag], :])).detach() * qf_loss_[:, :, [ag], :]
                        )
                    else:
                        qf_loss += (
                            torch.exp(torch.exp(log_pi - log_pi[:, :, [ag], :])).detach() * qf_loss_[:, :, [ag], :]
                        )

                qf_loss = qf_loss.mean()
            else:
                y_target = y_target.permute(0, 2, 1)
                qf_loss = self.qf_criterion(y_pred, y_target)

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
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Y Predictions",
                    total_y_pred,
                )
            )

        self._n_train_steps_total += 1
