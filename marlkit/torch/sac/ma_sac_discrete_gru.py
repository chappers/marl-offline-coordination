"""
This variation uses GRU or not, but operates batchwise
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.torch_rl_algorithm import MATorchTrainer


class SACTrainer(MATorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        plotter=None,
        render_eval_paths=False,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        # mac stuff
        mode="simple",
        use_shared_experience=False,
        use_central_critic=True,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.mode = mode
        self.use_shared_experience = use_shared_experience
        self.use_central_critic = use_central_critic

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        action_space_shape = (
            self.env.multi_agent_action_space.shape
            if hasattr(self.env, "multi_agent_action_space")
            else self.env.action_space.shape
        )

        if self.use_automatic_entropy_tuning:

            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(action_space_shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # since this is IAC paradigm, we can just stack everything and move on
        # since we're in the MA paradigm, we need to be careful of ragged
        # inputs...

        # print(batch.keys())
        # print(len(obs))
        # print(obs[0].shape)

        # as this is independent at this point in time, we can just concate obs
        # we only care later...
        if self.mode == "simple":
            """
            Policy and Alpha Loss
            """
            # calculate `action_prob` and `log_pi` over time
            # action_prob, log_pi, _ = self.policy(...)
            # simulatenously calculate qf1, qf2 as it becomes available so we only loop once
            batch_num = len(obs)
            action_probs = []
            log_pis = []
            new_action_probs = []
            new_log_pis = []
            for batch in range(batch_num):
                size = obs[batch].shape[1]
                path_len = obs[batch].shape[0]
                hidden = torch.cat(self.policy.init_hidden(size), 0)
                # qf1_hidden = torch.cat(self.qf1.init_hidden(size), 0)
                # qf2_hidden = torch.cat(self.qf2.init_hidden(size), 0)
                # qf1_p_hidden = torch.cat(self.qf1.init_hidden(size), 0)
                # qf2_p_hidden = torch.cat(self.qf2.init_hidden(size), 0)
                # target_qf1_hidden = torch.cat(self.target_qf1.init_hidden(size), 0)
                # target_qf2_hidden = torch.cat(self.target_qf2.init_hidden(size), 0)

                action_prob = []
                new_action_prob = []
                log_pi = []
                new_log_pi = []
                for t in range(path_len):

                    _, ap, logpi, _, hidden = self.policy(torch.from_numpy(obs[batch][t, :, :]).float(), hidden)
                    """
                    q1 = self.qf1(
                        *[torch.from_numpy(obs[batch][t, :, :]).float(), ap]
                    )
                    q2 = self.qf2(
                        *[torch.from_numpy(obs[batch][t, :, :]).float(), ap]
                    )
                    act_ = torch.from_numpy(actions[batch][t, :, :]).float()
                    q1_p = self.qf1(
                    *[torch.from_numpy(obs[batch][t, :, :]).float(), act_]
                    )
                    q2_p = self.qf2(
                        *[torch.from_numpy(obs[batch][t, :, :]).float(), act_]
                    )

                    # targets...
                    target_q1 = self.target_qf1(
                        *[torch.from_numpy(obs[batch][t, :, :]).float(), ap],
                    )
                    target_q2 = self.target_qf2(
                        *[torch.from_numpy(obs[batch][t, :, :]).float(), ap],
                    )
                    """
                    action_prob.append(ap)
                    log_pi.append(logpi)
                    """
                    q1_new_action.append(q1)
                    q2_new_action.append(q2)
                    q1_pred.append(q1_p)
                    q2_pred.append(q2_p)
                    """
                    if t != 0:
                        new_action_prob.append(ap)
                        new_log_pi.append(logpi)
                        """
                        target_q1_val.append(target_q1)
                        target_q2_val.append(target_q2)
                        """
                # do one more of new_action_prob
                _, ap, logpi, _, hidden = self.policy(torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden)
                """
                target_q1 = self.target_qf1(
                    *[torch.from_numpy(next_obs[batch][-1, :, :]).float(), ap],
                )
                target_q2 = self.target_qf2(
                    *[torch.from_numpy(next_obs[batch][-1, :, :]).float(), ap],
                )
                """
                new_action_prob.append(ap)
                new_log_pi.append(logpi)
                """
                target_q1_val.append(target_q1)
                target_q2_val.append(target_q2)
                """

                action_prob = torch.stack(action_prob, 0)
                new_action_prob = torch.stack(new_action_prob, 0)
                log_pi = torch.stack(log_pi, 0)
                new_log_pi = torch.stack(new_log_pi, 0)
                """
                q1_new_action = torch.stack(q1_new_action, 0)
                q2_new_action = torch.stack(q2_new_action, 0)
                q1_pred = torch.stack(q1_pred, 0)
                q2_pred = torch.stack(q2_pred, 0)
                target_q1_val = torch.stack(target_q1_val, 0)
                target_q2_val = torch.stack(target_q2_val, 0)
                """

                action_probs.append(action_prob)
                new_action_probs.append(new_action_prob)
                log_pis.append(log_pi)
                new_log_pis.append(new_log_pi)
                """
                q1_new_actions.append(q1_new_action)
                q2_new_actions.append(q2_new_action)
                q1_preds.append(q1_pred)
                q2_preds.append(q2_pred)
                target_q1_vals.append(target_q1_val)
                target_q2_vals.append(target_q2_val)
                """

            log_pis = torch.stack(log_pis, 0)
            action_probs = torch.stack(action_probs, 0)
            new_action_probs = torch.stack(new_action_probs, 0)
            new_log_pis = torch.stack(new_log_pis, 0)

            """
            q1_new_actions = torch.stack(q1_new_actions, 0)
            q2_new_actions = torch.stack(q2_new_actions, 0)
            q1_preds = torch.stack(q1_preds, 0)
            q2_preds = torch.stack(q2_preds, 0)
            target_q1_vals = torch.stack(target_q1_vals, 0)
            target_q2_vals = torch.stack(target_q2_vals, 0)
            """

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1

            # now calculate things off the q functions for the critic.
            q1_new_actions = self.qf1(torch.cat([torch.from_numpy(np.stack(obs, 0)).float(), action_probs], -1))
            q2_new_actions = self.qf2(torch.cat([torch.from_numpy(np.stack(obs, 0)).float(), action_probs], -1))
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)
            if self.use_shared_experience:
                # assume lambda = 1 as per paper, so we only need to iterate and not do the top part
                obs = torch.from_numpy(np.stack(obs, 0)).float()
                n_agents = obs.shape[-2]
                policy_loss_ = action_probs * (alpha * log_pis - q_new_actions)

                policy_loss = None
                for ag in range(n_agents):
                    # iterate through all of them...
                    if policy_loss is None:
                        policy_loss = (
                            torch.exp(torch.exp(log_pis - log_pis[:, :, [ag], :])) * policy_loss_[:, :, [ag], :]
                        )
                    else:
                        policy_loss += (
                            torch.exp(torch.exp(log_pis - log_pis[:, :, [ag], :])) * policy_loss_[:, :, [ag], :]
                        )
                policy_loss = policy_loss.mean()
            else:
                policy_loss = (action_probs * (alpha * log_pis - q_new_actions)).mean()
            # policy_loss = (alpha * log_pi - q_new_actions).mean()

            """
            QF Loss
            """
            # q1_preds, q2_preds, new_action_probs, new_log_pis, target_q1_vals, target_q2_vals
            target_q1_vals = self.target_qf1(
                torch.cat(
                    [torch.from_numpy(np.stack(next_obs, 0)).float(), new_action_probs],
                    -1,
                )
            )
            target_q2_vals = self.target_qf2(
                torch.cat(
                    [torch.from_numpy(np.stack(next_obs, 0)).float(), new_action_probs],
                    -1,
                )
            )
            target_q_values = new_action_probs * torch.min(target_q1_vals, target_q2_vals) - alpha * new_log_pis

            # update q_target
            n_action = target_q_values.shape[-1]
            rewards = torch.from_numpy(np.stack(rewards, axis=0)).float().permute(0, 1, 3, 2).repeat(1, 1, 1, n_action)
            terminals = (
                torch.from_numpy(np.stack(terminals, axis=0)).float().permute(0, 1, 3, 2).repeat(1, 1, 1, n_action)
            )

            q1_preds = self.qf1(
                torch.cat(
                    [
                        torch.from_numpy(np.stack(obs, 0)).float(),
                        torch.from_numpy(np.stack(actions, 0)).float(),
                    ],
                    -1,
                )
            )
            q2_preds = self.qf2(
                torch.cat(
                    [
                        torch.from_numpy(np.stack(obs, 0)).float(),
                        torch.from_numpy(np.stack(actions, 0)).float(),
                    ],
                    -1,
                )
            )

            q_target = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_q_values
        else:
            # calculate `action_prob` and `log_pi` over time
            # action_prob, log_pi, _ = self.policy(...)
            # simulatenously calculate qf1, qf2 as it becomes available so we only loop once
            batch_num = len(obs)
            action_probs = []
            log_pis = []
            q1_new_actions = []
            q2_new_actions = []
            q1_preds = []
            q2_preds = []
            target_q1_vals = []
            target_q2_vals = []
            new_action_probs = []
            new_log_pis = []
            for batch in range(batch_num):
                size = obs[batch].shape[1]
                path_len = obs[batch].shape[0]
                hidden = torch.cat(self.policy.init_hidden(size), 0)
                qf1_hidden = torch.cat(self.qf1.init_hidden(size), 0)
                qf2_hidden = torch.cat(self.qf2.init_hidden(size), 0)
                qf1_p_hidden = torch.cat(self.qf1.init_hidden(size), 0)
                qf2_p_hidden = torch.cat(self.qf2.init_hidden(size), 0)
                target_qf1_hidden = torch.cat(self.target_qf1.init_hidden(size), 0)
                target_qf2_hidden = torch.cat(self.target_qf2.init_hidden(size), 0)

                action_prob = []
                new_action_prob = []
                log_pi = []
                new_log_pi = []
                q1_new_action = []
                q2_new_action = []
                q1_pred = []
                q2_pred = []
                target_q1_val = []
                target_q2_val = []
                for t in range(path_len):

                    _, ap, logpi, _, hidden = self.policy(torch.from_numpy(obs[batch][t, :, :]).float(), hidden)
                    q1, qf1_hidden = self.qf1([torch.from_numpy(obs[batch][t, :, :]).float(), ap], qf1_hidden)
                    q2, qf2_hidden = self.qf2([torch.from_numpy(obs[batch][t, :, :]).float(), ap], qf2_hidden)
                    act_ = torch.from_numpy(actions[batch][t, :, :]).float()
                    q1_p, qf1_p_hidden = self.qf1(
                        [torch.from_numpy(obs[batch][t, :, :]).float(), act_],
                        qf1_p_hidden,
                    )
                    q2_p, qf2_p_hidden = self.qf2(
                        [torch.from_numpy(obs[batch][t, :, :]).float(), act_],
                        qf2_p_hidden,
                    )

                    # targets...
                    target_q1, target_qf1_hidden = self.target_qf1(
                        [torch.from_numpy(obs[batch][t, :, :]).float(), ap],
                        target_qf1_hidden,
                    )
                    target_q2, target_qf2_hidden = self.target_qf2(
                        [torch.from_numpy(obs[batch][t, :, :]).float(), ap],
                        target_qf2_hidden,
                    )
                    action_prob.append(ap)
                    log_pi.append(logpi)
                    q1_new_action.append(q1)
                    q2_new_action.append(q2)
                    q1_pred.append(q1_p)
                    q2_pred.append(q2_p)
                    if t != 0:
                        new_action_prob.append(ap)
                        new_log_pi.append(logpi)
                        target_q1_val.append(target_q1)
                        target_q2_val.append(target_q2)
                # do one more of new_action_prob
                _, ap, logpi, _, hidden = self.policy(torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden)
                target_q1, target_qf1_hidden = self.target_qf1(
                    [torch.from_numpy(next_obs[batch][-1, :, :]).float(), ap],
                    target_qf1_hidden,
                )
                target_q2, target_qf2_hidden = self.target_qf2(
                    [torch.from_numpy(next_obs[batch][-1, :, :]).float(), ap],
                    target_qf2_hidden,
                )
                new_action_prob.append(ap)
                new_log_pi.append(logpi)
                target_q1_val.append(target_q1)
                target_q2_val.append(target_q2)

                action_prob = torch.stack(action_prob, 0)
                new_action_prob = torch.stack(new_action_prob, 0)
                log_pi = torch.stack(log_pi, 0)
                new_log_pi = torch.stack(new_log_pi, 0)
                q1_new_action = torch.stack(q1_new_action, 0)
                q2_new_action = torch.stack(q2_new_action, 0)
                q1_pred = torch.stack(q1_pred, 0)
                q2_pred = torch.stack(q2_pred, 0)
                target_q1_val = torch.stack(target_q1_val, 0)
                target_q2_val = torch.stack(target_q2_val, 0)

                action_probs.append(action_prob)
                new_action_probs.append(new_action_prob)
                log_pis.append(log_pi)
                new_log_pis.append(new_log_pi)
                q1_new_actions.append(q1_new_action)
                q2_new_actions.append(q2_new_action)
                q1_preds.append(q1_pred)
                q2_preds.append(q2_pred)
                target_q1_vals.append(target_q1_val)
                target_q2_vals.append(target_q2_val)

            log_pis = torch.stack(log_pis, 0)
            q1_new_actions = torch.stack(q1_new_actions, 0)
            q2_new_actions = torch.stack(q2_new_actions, 0)
            q1_preds = torch.stack(q1_preds, 0)
            q2_preds = torch.stack(q2_preds, 0)
            action_probs = torch.stack(action_probs, 0)
            target_q1_vals = torch.stack(target_q1_vals, 0)
            target_q2_vals = torch.stack(target_q2_vals, 0)
            new_action_probs = torch.stack(new_action_probs, 0)
            new_log_pis = torch.stack(new_log_pis, 0)

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1

            # now calculate things off the q functions for the critic.
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)
            policy_loss = (action_probs * (alpha * log_pis - q_new_actions)).mean()
            # policy_loss = (alpha * log_pi - q_new_actions).mean()

            """
            QF Loss
            """
            # q1_preds, q2_preds, new_action_probs, new_log_pis, target_q1_vals, target_q2_vals

            target_q_values = new_action_probs * torch.min(target_q1_vals, target_q2_vals) - alpha * new_log_pis

            # update q_target
            n_action = target_q_values.shape[-1]
            rewards = torch.from_numpy(np.stack(rewards, axis=0)).float().permute(0, 1, 3, 2).repeat(1, 1, 1, n_action)
            terminals = (
                torch.from_numpy(np.stack(terminals, axis=0)).float().permute(0, 1, 3, 2).repeat(1, 1, 1, n_action)
            )

            q_target = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_q_values

        if self.use_shared_experience:
            # assume lambda = 1 as per paper, so we only need to iterate and not do the top part
            # otherwise we add additional loss when we have ag = ag style item or an identity matrix
            n_agents = obs.shape[-2]
            # policy_loss_ = (action_probs * (alpha * log_pis - q_new_actions))
            qf1_loss_ = (q1_preds - q_target.detach()) ** 2
            qf2_loss_ = (q2_preds - q_target.detach()) ** 2

            qf1_loss = None
            qf2_loss = None
            for ag in range(n_agents):
                # iterate through all of them...
                if qf1_loss is None:
                    qf1_loss = torch.exp(torch.exp(log_pis - log_pis[:, :, [ag], :])) * qf1_loss_[:, :, [ag], :]
                    qf2_loss = torch.exp(torch.exp(log_pis - log_pis[:, :, [ag], :])) * qf2_loss_[:, :, [ag], :]
                else:
                    qf1_loss += torch.exp(torch.exp(log_pis - log_pis[:, :, [ag], :])) * qf1_loss_[:, :, [ag], :]
                    qf2_loss += torch.exp(torch.exp(log_pis - log_pis[:, :, [ag], :])) * qf2_loss_[:, :, [ag], :]
            qf1_loss = qf1_loss.mean()
            qf2_loss = qf2_loss.mean()
        else:
            qf1_loss = self.qf_criterion(q1_preds, q_target.detach())
            qf2_loss = self.qf_criterion(q2_preds, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        if self.use_shared_experience or self.use_central_critic:
            qf1_loss.backward(retain_graph=True)
        else:
            qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        if self.use_shared_experience or self.use_central_critic:
            qf2_loss.backward(retain_graph=True)
        else:
            qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (action_probs * (log_pis - q_new_actions)).mean()
            # policy_loss = (alpha * log_pi - q_new_actions).mean()
            # policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_preds),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_preds),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    ptu.get_numpy(q_target),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pis),
                )
            )
            if self.use_automatic_entropy_tuning:
                self.eval_statistics["Alpha"] = alpha.item()
                self.eval_statistics["Alpha Loss"] = alpha_loss.item()

            try:
                self.eval_statistics["env_count"] = self.env._env_count
                self.env.set_switch_progress(self._n_train_steps_total / 249000)
            except:
                pass
            self.eval_statistics["n_train_steps_total"] = self._n_train_steps_total
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
