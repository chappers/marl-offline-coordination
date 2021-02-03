"""
This variation just uses gumbel-softmax extension to get discrete actions.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.torch_rl_algorithm import MATorchTrainer


class DDPGTrainer(MATorchTrainer):
    """
    Deep Deterministic Policy Gradient
    """

    def __init__(
        self,
        qf,
        target_qf,
        policy,
        target_policy,
        discount=0.99,
        reward_scale=1.0,
        policy_learning_rate=1e-4,
        qf_learning_rate=1e-3,
        qf_weight_decay=0,
        target_hard_update_period=1000,
        tau=1e-2,
        use_soft_update=False,
        qf_criterion=None,
        policy_pre_activation_weight=0.0,
        optimizer_class=optim.Adam,
        min_q_value=-np.inf,
        max_q_value=np.inf,
        # mac stuff
        use_shared_experience=False,
        use_joint_space=False,  # for MADDPG
        state_dim=None,
        n_agents=None,
        n_actions=None,
        mrl=False,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf = qf
        self.target_qf = target_qf
        self.policy = policy
        self.target_policy = target_policy
        self.use_shared_experience = use_shared_experience
        self.use_joint_space = use_joint_space
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.mrl = mrl

        self.discount = discount
        self.reward_scale = reward_scale

        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.qf_criterion = qf_criterion
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        states = batch["states"]
        next_states = batch["next_states"]

        # since this is IPG paradigm, we can just stack everything and move on
        # since we're in the MA paradigm, we need to be careful of ragged
        # inputs...
        """
        obs = torch.from_numpy(np.stack(obs, 0)).float()
        actions = torch.from_numpy(np.stack(actions, 0)).float()
        terminals = torch.from_numpy(np.stack(terminals, 0)).float()
        rewards = torch.from_numpy(np.stack(rewards, 0)).float()
        states = torch.from_numpy(np.stack(states, 0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, 0)).float()
        next_states = torch.from_numpy(np.stack(next_states, 0)).float()

        terminals = terminals.permute(0, 1, 3, 2)
        rewards = rewards.permute(0, 1, 3, 2)
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

        # statistics
        total_qf_loss = []
        total_policy_loss = []
        total_raw_policy_loss = []
        total_q_pred = []
        total_q_target = []
        total_bellman_errors = []
        total_policy_actions = []

        for b in range(len(obs)):
            try:
                rewards = to_tensor(batch["rewards"][b])
                terminals = to_tensor(batch["terminals"][b])
                obs = to_tensor(batch["observations"][b])
                states = to_tensor(batch["states"][b])
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
                states = to_tensor(batch["states"][b], filter_n)
                active_agent = to_tensor(batch["active_agents"][b], filter_n)
                # state_0 = batch["states_0"]
                actions = to_tensor(batch["actions"][b], filter_n)
                next_obs = to_tensor(batch["next_observations"][b], filter_n)
                next_states = to_tensor(batch["next_states"][b], filter_n)

            terminals = terminals.permute(0, 2, 1)
            rewards = rewards.permute(0, 2, 1)

            """
            Policy operations.
            """
            if self.policy_pre_activation_weight > 0:
                raise NotImplementedError
                """
                policy_actions, pre_tanh_value = self.policy(
                    obs,
                    return_preactivations=True,
                )
                pre_activation_policy_loss = (pre_tanh_value ** 2).sum(dim=1).mean()
                q_output = self.qf(obs, policy_actions)
                raw_policy_loss = -q_output.mean()
                policy_loss = raw_policy_loss + pre_activation_policy_loss * self.policy_pre_activation_weight
                """
            else:
                policy_actions = self.policy(obs)
                if self.use_joint_space:
                    n_agents = policy_actions.shape[-2]
                    # if n_agents != n_agents:
                    #    # pad or repeat until?
                    # print("policy_actions", policy_actions.shape)
                    # print("states", states.shape)
                    rep_policy_actions = policy_actions.detach().repeat(1, 1, n_agents)
                    rep_states = states.detach().repeat(1, n_agents, 1)

                    # print("rep_policy_actions", rep_policy_actions.shape)  # this needs to be resize or padded

                    #
                    if self.n_actions is not None and self.n_agents is not None:
                        target_action_agent = self.n_actions * self.n_agents
                        if target_action_agent != rep_policy_actions.size(2):
                            pad_target = (target_action_agent - rep_policy_actions.size(2)) // 2
                            rep_policy_actions = nn.ReplicationPad1d(
                                (pad_target, target_action_agent - rep_policy_actions.size(2) - pad_target)
                            )(rep_policy_actions)
                    if self.state_dim is not None:
                        if rep_states.size(2) != self.state_dim:
                            pad_target = (self.state_dim - rep_states.size(2)) // 2
                            rep_states = nn.ReplicationPad1d(
                                (pad_target, self.state_dim - pad_target - rep_states.size(2))
                            )(rep_states)
                    flat_inputs = torch.cat([obs, policy_actions, rep_policy_actions, rep_states], dim=-1)
                    if self.n_agents is not None:
                        if n_agents != self.n_agents:
                            pad_target = (self.n_agents - n_agents) // 2
                            flat_inputs = flat_inputs.permute(0, 2, 1)
                            flat_inputs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                                flat_inputs
                            )
                            flat_inputs = flat_inputs.permute(0, 2, 1)
                else:
                    flat_inputs = torch.cat([obs, policy_actions], dim=-1)
                q_output = self.qf(flat_inputs)
                raw_policy_loss = policy_loss = -q_output.mean()

            """
            Critic operations.
            """
            next_actions = self.target_policy(next_obs)
            # speed up computation by not backpropping these gradients
            next_actions.detach()
            if self.use_joint_space:
                n_agents = next_actions.shape[-2]
                rep_next_actions = next_actions.repeat(1, 1, n_agents)
                rep_next_states = next_states.detach().repeat(1, n_agents, 1)

                if self.n_actions is not None and self.n_agents is not None:
                    target_action_agent = self.n_actions * self.n_agents
                    if target_action_agent != rep_next_actions.size(2):
                        pad_target = (target_action_agent - rep_next_actions.size(2)) // 2
                        rep_next_actions = nn.ReplicationPad1d(
                            (pad_target, target_action_agent - rep_next_actions.size(2) - pad_target)
                        )(rep_next_actions)
                if self.n_actions is not None and self.n_agents is not None:
                    target_action_agent = self.n_actions * self.n_agents
                    if target_action_agent != rep_next_actions.size(2):
                        pad_target = (target_action_agent - rep_next_actions.size(2)) // 2
                        rep_next_actions = nn.ReplicationPad1d(
                            (pad_target, target_action_agent - rep_next_actions.size(2) - pad_target)
                        )(rep_next_actions)
                if self.state_dim is not None:
                    if rep_next_states.size(2) != self.state_dim:
                        pad_target = (self.state_dim - rep_next_states.size(2)) // 2
                        rep_next_states = nn.ReplicationPad1d(
                            (pad_target, self.state_dim - pad_target - rep_next_states.size(2))
                        )(rep_next_states)
                flat_inputs = torch.cat([next_obs, next_actions, rep_next_actions, rep_next_states], dim=-1)
                if self.n_agents is not None:
                    if n_agents != self.n_agents:
                        pad_target = (self.n_agents - n_agents) // 2
                        flat_inputs = flat_inputs.permute(0, 2, 1)
                        flat_inputs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                            flat_inputs
                        )
                        flat_inputs = flat_inputs.permute(0, 2, 1)
            else:
                flat_inputs = torch.cat([next_obs, next_actions], -1)
            target_q_values = self.target_qf(flat_inputs)
            if self.n_agents is not None:
                n_agents = rewards.size(1)
                if n_agents != self.n_agents:
                    pad_target = (self.n_agents - n_agents) // 2
                    rewards = rewards.permute(0, 2, 1)
                    terminals = terminals.permute(0, 2, 1)
                    rewards = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(rewards)
                    terminals = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(terminals)
                    rewards = rewards.permute(0, 2, 1)
                    terminals = terminals.permute(0, 2, 1)

            if self.mrl:
                # augment rewards
                mrl_log_proba = torch.log(torch.max(self.policy(obs).softmax(-1), -1)[0])
                mrl_log_proba = mrl_log_proba[:, torch.randperm(mrl_log_proba.size(-1))]
                rewards = rewards + mrl_log_proba.unsqueeze(-1)

            q_target = rewards + (1.0 - terminals) * self.discount * target_q_values
            q_target = q_target.detach()
            q_target = torch.clamp(q_target, self.min_q_value, self.max_q_value)
            if self.use_joint_space:
                n_agents = actions.shape[-2]
                rep_actions = actions.repeat(1, 1, n_agents)
                rep_states = states.repeat(1, n_agents, 1)
                if self.state_dim is not None:
                    if rep_states.size(2) != self.state_dim:
                        pad_target = (self.state_dim - rep_states.size(2)) // 2
                        rep_states = nn.ReplicationPad1d(
                            (pad_target, self.state_dim - pad_target - rep_states.size(2))
                        )(rep_states)
                flat_inputs = torch.cat([obs, policy_actions, rep_policy_actions, rep_states], dim=-1)
                if self.n_agents is not None:
                    if n_agents != self.n_agents:
                        pad_target = (self.n_agents - n_agents) // 2
                        flat_inputs = flat_inputs.permute(0, 2, 1)
                        flat_inputs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                            flat_inputs
                        )
                        flat_inputs = flat_inputs.permute(0, 2, 1)
            else:
                flat_inputs = torch.cat([obs, actions], -1)
            q_pred = self.qf(flat_inputs)
            bellman_errors = (q_pred - q_target) ** 2
            raw_qf_loss = self.qf_criterion(q_pred, q_target)

            if self.qf_weight_decay > 0:
                reg_loss = self.qf_weight_decay * sum(
                    torch.sum(param ** 2) for param in self.qf.regularizable_parameters()
                )
                qf_loss = raw_qf_loss + reg_loss
            else:
                qf_loss = raw_qf_loss

            """
            Update Networks
            """

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()

            self.qf_optimizer.zero_grad()
            qf_loss.backward(retain_graph=True)
            self.qf_optimizer.step()

            self._update_target_networks()

            total_qf_loss.append(ptu.get_numpy(qf_loss))
            total_policy_loss.append(ptu.get_numpy(policy_loss))
            total_raw_policy_loss.append(ptu.get_numpy(raw_policy_loss))
            total_q_pred.append(np.mean(ptu.get_numpy(q_pred)))
            total_q_target.append(np.mean(ptu.get_numpy(q_target)))
            total_bellman_errors.append(np.mean(ptu.get_numpy(bellman_errors)))
            total_policy_actions.append(np.mean(ptu.get_numpy(policy_actions)))
        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(total_qf_loss)
            self.eval_statistics["Policy Loss"] = np.mean(total_policy_loss)
            self.eval_statistics["Raw Policy Loss"] = np.mean(total_raw_policy_loss)
            self.eval_statistics["Preactivation Policy Loss"] = (
                self.eval_statistics["Policy Loss"] - self.eval_statistics["Raw Policy Loss"]
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Predictions",
                    np.stack(total_q_pred, 0),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    np.stack(total_q_target, 0),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Bellman Errors",
                    np.stack(total_bellman_errors, 0),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy Action",
                    np.stack(total_policy_actions, 0),
                )
            )
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)
        else:
            if self._n_train_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_policy,
            self.target_qf,
        ]

    def get_epoch_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
            trained_policy=self.policy,
            target_policy=self.target_policy,
        )
