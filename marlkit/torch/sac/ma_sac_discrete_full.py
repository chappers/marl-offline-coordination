"""
This variation does not use GRU, but is used when things are shared.
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
        use_shared_experience=False,
        use_central_critic=False,
        n_agents=None,
        state_dim=None,
        mrl=False,
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
        self.use_shared_experience = use_shared_experience
        self.use_central_critic = use_central_critic
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mrl = mrl

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
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # since this is IAC paradigm, we can just stack everything and move on
        # since we're in the MA paradigm, we need to be careful of ragged
        """
        obs = torch.from_numpy(np.stack(obs, 0)).float()
        actions = torch.from_numpy(np.stack(actions, 0)).float()
        states = torch.from_numpy(np.stack(states, 0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, 0)).float()
        next_states = torch.from_numpy(np.stack(next_states, 0)).float()
        """

        def to_tensor(x):
            try:
                return torch.from_numpy(np.array(x, dtype=float)).float()
            except:
                x = np.stack([np.array(x_).flatten()[np.newaxis, :] for x_ in x], 0)
                return torch.from_numpy(x).float()

        # statistics
        total_qf1_loss = []
        total_qf2_loss = []
        total_q1_preds = []
        total_q2_preds = []
        total_q_target = []
        total_policy_loss = []

        for b in range(len(obs)):
            rewards = to_tensor(batch["rewards"][b]).unsqueeze(0)
            terminals = to_tensor(batch["terminals"][b]).unsqueeze(0)
            obs = to_tensor(batch["observations"][b]).unsqueeze(0)
            states = to_tensor(batch["states"][b]).unsqueeze(0)
            next_states = to_tensor(batch["next_states"][b]).unsqueeze(0)
            active_agent = to_tensor(batch["active_agents"][b]).unsqueeze(0)
            # state_0 = batch["states_0"]
            actions = to_tensor(batch["actions"][b]).unsqueeze(0)
            next_obs = to_tensor(batch["next_observations"][b]).unsqueeze(0)

            # print(batch.keys())
            # print(len(obs))
            # print(obs[0].shape)

            # as this is independent at this point in time, we can just concate obs
            # we only care later...
            """
            Policy and Alpha Loss
            """
            # calculate `action_prob` and `log_pi` over time
            # action_prob, log_pi, _ = self.policy(...)
            # simulatenously calculate qf1, qf2 as it becomes available so we only loop once
            batch_num = len(obs)
            _, action_probs, log_pis, _ = self.policy(
                obs,
                reparameterize=True,
                return_log_prob=True,
            )

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
            if self.use_central_critic:
                n_agents = action_probs.shape[-2]
                # pad this to self.n_agents
                # if n_agents != self.n_agents:
                #    action_probs = action_probs.squeeze(0)
                #    action_probs = action_probs.permute(0, 2, 1)
                #    pad_target = (self.n_agents - n_agents)//2
                #    action_probs = nn.ReplicationPad1d((pad_target, self.n_agents-pad_target-n_agents))(action_probs)
                #    action_probs = action_probs.permute(0, 2, 1)
                #    action_probs = action_probs.unsqueeze(0)

                # pad states.
                if self.state_dim is not None:
                    if states.size(2) != self.state_dim:
                        pad_target = (self.state_dim - states.size(2)) // 2
                        states = states.squeeze(0)
                        states = nn.ReplicationPad1d((pad_target, self.state_dim - pad_target - states.size(2)))(states)
                        states = states.unsqueeze(0)

                flat_inputs = torch.cat([states.repeat(1, 1, n_agents, 1), action_probs], -1)
                # print(flat_inputs.shape)
                # print(flat_inputs.shape, self.qf1.input_size)
                if self.n_agents is not None:
                    if n_agents != self.n_agents:
                        pad_target = (self.n_agents - n_agents) // 2
                        flat_inputs = flat_inputs.squeeze(0)
                        flat_inputs = flat_inputs.permute(0, 2, 1)
                        flat_inputs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                            flat_inputs
                        )
                        flat_inputs = flat_inputs.permute(0, 2, 1)
                        flat_inputs = flat_inputs.unsqueeze(0)

                # pad by n_agents
                q1_new_actions = self.qf1(flat_inputs)
                q2_new_actions = self.qf2(flat_inputs)
            else:
                q1_new_actions = self.qf1(torch.cat([obs, action_probs], -1))
                q2_new_actions = self.qf2(torch.cat([obs, action_probs], -1))
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)

            if self.use_shared_experience:
                # assume lambda = 1 as per paper, so we only need to iterate and not do the top part
                n_agents = obs.shape[-2]
                policy_loss_ = action_probs * (alpha * log_pis - q_new_actions)

                policy_loss = None
                for ag in range(n_agents):
                    # iterate through all of them...
                    if policy_loss is None:
                        policy_loss = (
                            torch.exp(log_pis - log_pis[:, :, [ag], :]).detach() * policy_loss_[:, :, [ag], :]
                        ) * 0.1
                    else:
                        policy_loss += (
                            torch.exp(log_pis - log_pis[:, :, [ag], :]).detach() * policy_loss_[:, :, [ag], :]
                        ) * 0.1
                policy_loss = policy_loss.mean()
            else:
                n_agents = action_probs.size(2)
                if self.n_agents is not None:
                    if n_agents != self.n_agents:
                        pad_target = (self.n_agents - n_agents) // 2
                        action_probs = action_probs.squeeze(0).permute(0, 2, 1)
                        log_pis = log_pis.squeeze(0).permute(0, 2, 1)
                        action_probs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                            action_probs
                        )
                        log_pis = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(log_pis)
                        action_probs = action_probs.permute(0, 2, 1).unsqueeze(0)
                        log_pis = log_pis.permute(0, 2, 1).unsqueeze(0)

                policy_loss = (action_probs * (alpha * log_pis - q_new_actions)).mean()

            """
            QF Loss
            """
            if self.use_central_critic:
                n_agents = action_probs.shape[-2]
                q1_preds = self.qf1(torch.cat([states.repeat(1, 1, n_agents, 1), action_probs], -1))
                q2_preds = self.qf2(torch.cat([states.repeat(1, 1, n_agents, 1), action_probs], -1))
            else:
                q1_preds = self.qf1(torch.cat([obs, actions], -1))
                q2_preds = self.qf2(torch.cat([obs, actions], -1))

            _, new_action_probs, new_log_pis, _ = self.policy(
                next_obs,
                reparameterize=True,
                return_log_prob=True,
            )

            # q1_preds, q2_preds, new_action_probs, new_log_pis, target_q1_vals, target_q2_vals
            if self.use_central_critic:
                n_agents = new_action_probs.shape[-2]
                if self.state_dim is not None:
                    if next_states.size(2) != self.state_dim:
                        pad_target = (self.state_dim - next_states.size(2)) // 2
                        next_states = next_states.squeeze(0)
                        next_states = nn.ReplicationPad1d(
                            (pad_target, self.state_dim - pad_target - next_states.size(2))
                        )(next_states)
                        next_states = next_states.unsqueeze(0)
                flat_inputs = torch.cat([next_states.repeat(1, 1, n_agents, 1), new_action_probs], -1)
                if self.n_agents is not None:
                    if n_agents != self.n_agents:
                        pad_target = (self.n_agents - n_agents) // 2
                        flat_inputs = flat_inputs.squeeze(0)
                        flat_inputs = flat_inputs.permute(0, 2, 1)
                        flat_inputs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                            flat_inputs
                        )
                        flat_inputs = flat_inputs.permute(0, 2, 1).unsqueeze(0)
                target_q1_vals = self.target_qf1(flat_inputs)
                target_q2_vals = self.target_qf2(flat_inputs)
            else:
                target_q1_vals = self.target_qf1(torch.cat([next_obs, new_action_probs], -1))
                target_q2_vals = self.target_qf2(torch.cat([next_obs, new_action_probs], -1))

            if self.n_agents is not None:
                if n_agents != self.n_agents:
                    pad_target = (self.n_agents - n_agents) // 2
                    new_action_probs = new_action_probs.squeeze(0).permute(0, 2, 1)
                    new_log_pis = new_log_pis.squeeze(0).permute(0, 2, 1)
                    new_action_probs = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(
                        new_action_probs
                    )
                    new_log_pis = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(new_log_pis)
                    new_action_probs = new_action_probs.permute(0, 2, 1).unsqueeze(0)
                    new_log_pis = new_log_pis.permute(0, 2, 1).unsqueeze(0)
            target_q_values = new_action_probs * torch.min(target_q1_vals, target_q2_vals) - alpha * new_log_pis

            # update q_target
            n_action = target_q_values.shape[-1]
            rewards = torch.from_numpy(np.stack(rewards, axis=0)).float().permute(0, 1, 3, 2).repeat(1, 1, 1, n_action)
            terminals = (
                torch.from_numpy(np.stack(terminals, axis=0)).float().permute(0, 1, 3, 2).repeat(1, 1, 1, n_action)
            )
            if self.mrl:
                # the munchausen RL augmentation in this setting is to regularise with other agent policies and not itself...
                mrl_log_proba = self.policy.get_log_proba(obs)
                # - print(mrl_log_proba.shape, rewards.shape)
                # - mrl_log_proba = torch.max(mrl_log_proba, -1, keepdims=True)[0]
                # do shuffle
                mrl_log_proba = mrl_log_proba[:, :, torch.randperm(mrl_log_proba.size(-2))]
                rewards = rewards + mrl_log_proba

            # print(rewards.shape, terminals.shape, target_q_values.shape)
            if self.n_agents is not None:
                n_agents = rewards.size(2)
                if n_agents != self.n_agents:
                    pad_target = (self.n_agents - n_agents) // 2
                    rewards = rewards.squeeze(0).permute(0, 2, 1)
                    terminals = terminals.squeeze(0).permute(0, 2, 1)
                    rewards = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(rewards)
                    terminals = nn.ReplicationPad1d((pad_target, self.n_agents - pad_target - n_agents))(terminals)
                    rewards = rewards.permute(0, 2, 1).unsqueeze(0)
                    terminals = terminals.permute(0, 2, 1).unsqueeze(0)
            q_target = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_q_values

            if self.use_shared_experience:
                # assume lambda = 1 as per paper, so we only need to iterate and not do the top part
                n_agents = obs.shape[-2]
                # policy_loss_ = (action_probs * (alpha * log_pis - q_new_actions))
                qf1_loss_ = (q1_preds - q_target.detach()) ** 2
                qf2_loss_ = (q2_preds - q_target.detach()) ** 2

                qf1_loss = None
                qf2_loss = None
                for ag in range(n_agents):
                    # iterate through all of them...
                    if qf1_loss is None:
                        qf1_loss = torch.exp(log_pis - log_pis[:, :, [ag], :]).detach() * qf1_loss_[:, :, [ag], :]
                        qf2_loss = torch.exp(log_pis - log_pis[:, :, [ag], :]).detach() * qf2_loss_[:, :, [ag], :]
                    else:
                        qf1_loss += torch.exp(log_pis - log_pis[:, :, [ag], :]).detach() * qf1_loss_[:, :, [ag], :]
                        qf2_loss += torch.exp(log_pis - log_pis[:, :, [ag], :]).detach() * qf2_loss_[:, :, [ag], :]

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

            total_qf1_loss.append(ptu.get_numpy(qf1_loss))
            total_qf2_loss.append(ptu.get_numpy(qf2_loss))
            total_q1_preds.append(np.mean(ptu.get_numpy(q1_preds)))
            total_q2_preds.append(np.mean(ptu.get_numpy(q2_preds)))
            total_q_target.append(np.mean(ptu.get_numpy(q_target)))
            total_policy_loss.append(ptu.get_numpy(policy_loss))

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
            # policy_loss = (action_probs * (log_pis - q_new_actions)).mean()
            # policy_loss = (alpha * log_pi - q_new_actions).mean()
            # policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(total_qf1_loss)
            self.eval_statistics["QF2 Loss"] = np.mean(total_qf2_loss)
            self.eval_statistics["Policy Loss"] = np.mean(total_policy_loss)
            # self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    np.stack(total_q1_preds, 0),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    np.stack(total_q2_preds, 0),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    np.stack(total_q_target, 0),
                )
            )
            # self.eval_statistics.update(
            #    create_stats_ordered_dict(
            #        "Log Pis",
            #        ptu.get_numpy(log_pis),
            #    )
            # )
            # if self.use_automatic_entropy_tuning:
            #    self.eval_statistics["Alpha"] = alpha.item()
            #    self.eval_statistics["Alpha Loss"] = alpha_loss.item()

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
