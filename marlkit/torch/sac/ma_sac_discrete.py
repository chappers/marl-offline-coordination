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
        use_shared_experience=False,
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
        if use_shared_experience:
            raise Exception("Shared Experience works only with FULL replay buffer")
        self.use_shared_experience = use_shared_experience

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            action_space_shape = (
                self.env.multi_agent_action_space.shape
                if hasattr(self.env, "multi_agent_action_space")
                else self.env.action_space.shape
            )
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

        """
        Policy and Alpha Loss
        """
        # no need to worry about groups of games. in the IAC setting.
        obs = torch.from_numpy(np.concatenate(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.concatenate(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.concatenate(terminals, axis=0)).float()
        actions = torch.from_numpy(np.concatenate(actions, axis=0)).float()
        rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).float()

        _, action_prob, log_pi, _ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        # q_new_actions = torch.min(
        #     self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions),
        # )
        q_new_actions = torch.min(
            self.qf1(obs, action_prob),
            self.qf2(obs, action_prob),
        )

        # SEAC uses lambda = 1...
        policy_loss = (action_prob * (alpha * log_pi - q_new_actions)).mean()

        """
        QF Loss
        """
        # q1_pred = self.qf1(obs, actions)
        # q2_pred = self.qf2(obs, actions)

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        _, new_action_prob, new_log_pi, _ = self.policy(
            next_obs,
            reparameterize=True,
            return_log_prob=True,
        )

        # something like this `min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi`
        # target_q_values = (
        #     torch.min(
        #         self.target_qf1(next_obs, new_next_actions),
        #         self.target_qf2(next_obs, new_next_actions),
        #     )
        #     - alpha * new_log_pi
        # )

        # new_action_prob = 0  # TODO update this from self.policy
        target_q_values = (
            new_action_prob
            * torch.min(
                self.target_qf1(next_obs, new_action_prob),
                self.target_qf2(next_obs, new_action_prob),
            )
            - alpha * new_log_pi
        )

        rewards = rewards.reshape(-1, 1).float()
        terminals = terminals.reshape(-1, 1).float()

        q_target = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_q_values

        # value loss function
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
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
            policy_loss = (action_prob * (log_pi - q_new_actions)).mean()
            # policy_loss = (alpha * log_pi - q_new_actions).mean()
            # policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
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
                    ptu.get_numpy(log_pi),
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
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
