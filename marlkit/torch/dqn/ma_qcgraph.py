"""
An implementation of Qtran base only as per pymarl
"""

import numpy as np
import torch
from torch import nn as nn

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer
from marlkit.policies.argmax import MAArgmaxDiscretePolicy
import torch.optim as optim
from collections import OrderedDict


class DoubleDQNTrainer(DQNTrainer):
    opt_loss = 1
    nopt_min_loss = 0.1

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        state = batch["states"]
        active_agent = batch["active_agents"]
        # state_0 = batch["states_0"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        n_agents = None,
        state_dim = None,
        action_dim = None,
        obs_dim = None,
        """

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

            obs_qf, hidden_states = self.qf(next_obs, return_hidden=True)
            best_action_idxs = obs_qf.max(-1, keepdim=True)[1]
            # print(best_action_idxs.shape)
            # print(self.target_qf(next_obs).shape)
            next_obs_qf, target_hidden_states = self.target_qf(next_obs, return_hidden=True)
            # target_best_action_idxs = next_obs_qf.max(-1, keepdim=True)[1]
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
                # we expect a qtran mixer here!
                """
                y_pred = y_pred.permute(0, 2, 1)  # needs to match y_pred size
                # we need to pad out y_pred with agent active?
                if y_pred.shape != active_agent.shape:
                    # need to concate along 0 axis...
                    pad_y_pred_shape = list(active_agent.shape)
                    pad_y_pred_shape[-1] = active_agent.shape[-1] - y_pred.shape[-1]
                    y_pred = torch.cat([y_pred, torch.zeros(*pad_y_pred_shape)], axis=-1)
                    y_target = torch.cat([y_target, torch.zeros(*pad_y_pred_shape)], axis=-1)
                """
                # pad state!
                # print("state", state.shape)

                max_actions = torch.zeros(size=(obs.shape[0], obs.shape[1], actions.shape[-1]))
                max_actions_onehot = max_actions.scatter(-1, best_action_idxs[:, :], 1)
                joint_qs, vs = self.mixer(obs, actions, state, hidden_states)
                target_joint_qs, target_vs = self.target_mixer(
                    next_obs, max_actions_onehot, next_states, target_hidden_states
                )

                # Td loss targets
                gamma = 0.99
                if target_joint_qs.size(1) != 1:
                    # ensure consistency
                    target_joint_qs = torch.mean(target_joint_qs, 1, keepdim=True)
                    joint_qs = torch.mean(joint_qs, 1, keepdim=True)
                    vs = torch.mean(vs, 1, keepdim=True)

                td_targets = torch.mean(rewards, 2) + gamma * (1 - torch.mean(terminals, 2)) * target_joint_qs
                td_error = joint_qs - td_targets.detach()
                td_loss = (td_error ** 2).sum()
                # -- TD Loss --

                # -- Opt Loss --
                max_joint_qs, _ = self.mixer(
                    obs, max_actions_onehot, state, hidden_states
                )  # Don't use the target network and target agent max actions as per author's email
                if max_joint_qs.size(1) != 1:
                    max_joint_qs = torch.mean(max_joint_qs, 1, keepdim=True)

                # max_actions_qvals is the best joint-action computed by agents
                max_actions_qvals, max_actions_current = obs_qs.max(dim=-1, keepdim=True)
                opt_error = max_actions_qvals.sum(dim=1) - max_joint_qs.detach() + vs
                opt_loss = (opt_error ** 2).sum()
                # -- Opt Loss --

                # -- Nopt Loss --
                # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
                chosen_action_qvals = torch.gather(obs_qs, dim=2, index=actions.long())

                # Don't use target networks here either
                # print(chosen_action_qvals.shape, joint_qs.shape, vs.shape)
                if joint_qs.size(-1) != 1:
                    joint_qs = torch.mean(joint_qs, dim=-1, keepdim=True)
                nopt_values = chosen_action_qvals.sum(dim=1) - joint_qs.detach() + vs
                nopt_error = nopt_values.clamp(max=0)
                nopt_loss = (nopt_error ** 2).sum()
                # -- Nopt loss --

                qf_loss = td_loss + self.opt_loss * opt_loss + self.nopt_min_loss * nopt_loss

            else:
                raise NotImplementedError

            """
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
