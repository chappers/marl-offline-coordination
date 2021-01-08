"""
An implementation of Mixer variation just to familiarise myself with using this custom setup. 
We'll assume the usage of the RNN agent as well, so we'll need to handle the hidden states here
"""

import numpy as np
import torch
import torch.nn as nn

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer
import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.torch_rl_algorithm import MATorchTrainer
from marlkit.policies.argmax import MAArgmaxDiscretePolicy
import torch.optim as optim
from collections import OrderedDict


class DoubleDQNTrainer(DQNTrainer):
    def calculate_qf(self, obs):
        """
        returns the output of the qf with RNN, or otherwise
        see here:
        https://github.com/ray-project/ray/blob/master/rllib/agents/qmix/model.py
        """
        if hasattr(self.qf, "init_hidden"):
            h = self.qf.init_hidden()
            agent_out = []
            max_seq_len = obs.shape[-1]
            for t in range(max_seq_len):
                # do stuff now.
                q, h = self.qf.forward(obs[:, t], h)
                agent_out.append(q)
            return torch.stack(agent_out, dim=1)
        else:
            return self.qf(obs_item)

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        state = batch["states"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # in the mixer setting they need to be managed in groups
        size = obs[0].shape[0]
        path_len = obs[0].shape[-1]
        batch_num = len(obs)

        # everything revolves around whole paths when using GRU

        """
        Compute loss
        """
        # compute: best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
        # this is "equivalent" to self.qf(next_obs) and self.qf(obs)
        best_action_idxs = []
        obs_qs = []
        for batch in range(batch_num):
            size = obs[batch].shape[1]
            path_len = obs[batch].shape[0]
            hidden = torch.cat(self.qf.init_hidden(size), 0)
            best_action_idx = []
            obs_q = []
            for t in range(path_len):
                q_ = []
                # for agent_indx in range(size):
                #    print(torch.from_numpy(obs[batch][t, agent_indx, :]).float().shape)
                #    print(hidden[agent_indx].float().shape)
                #    q, h = self.qf(torch.from_numpy(obs[batch][[t], agent_indx, :]).float(), hidden[agent_indx].float())
                #    hidden[agent_indx] = h
                #    q_.append(q)

                q, hidden = self.qf(
                    torch.from_numpy(obs[batch][t, :, :]).float(), hidden
                )
                if t != 0:
                    best_action_idx.append(q)
                obs_q.append(q)

            q, hidden = self.qf(
                torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden
            )
            best_action_idx.append(q)
            best_action_idx = torch.stack(best_action_idx, 0)
            obs_q = torch.stack(obs_q, 0)
            # if self.mixer is None:
            if True:
                best_action_idx = best_action_idx.max(-1, keepdim=True)[1]
            best_action_idxs.append(best_action_idx.permute(0, 2, 1))
            obs_qs.append(obs_q)

        best_action_idxs = torch.stack(best_action_idxs, 0)
        obs_qs = torch.stack(obs_qs, 0)

        # compute: target_q_values = self.target_qf(next_obs).gather(1, best_action_idxs).detach()
        target_q_values = []
        with torch.no_grad():
            for batch in range(batch_num):
                size = obs[batch].shape[1]
                path_len = obs[batch].shape[0]
                hidden = torch.cat(self.target_qf.init_hidden(size), 0)
                target_q_value = []
                for t in range(path_len):
                    q_ = []
                    q, hidden = self.target_qf(
                        torch.from_numpy(obs[batch][t, :, :]).float(), hidden
                    )
                    if t != 0:
                        target_q_value.append(q)

                q, hidden = self.qf(
                    torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden
                )
                target_q_value.append(q)
                target_q_value = torch.stack(target_q_value, 0)
                # if self.mixer is None:
                if True:
                    target_q_value = target_q_value.max(-1, keepdim=True)[1]
                target_q_values.append(target_q_value.permute(0, 2, 1))

        # need to gather..., by best_action_idxs
        target_q_values = torch.stack(target_q_values, 0)
        y_target = (
            torch.from_numpy(np.stack(rewards, axis=0)).float()
            + (1.0 - torch.from_numpy(np.stack(terminals, axis=0)).float())
            * self.discount
            * target_q_values
        )
        y_target = y_target.detach()
        y_target = y_target.permute(0, 1, 3, 2)

        # actions is a one-hot vector
        y_pred = torch.sum(
            obs_qs * torch.from_numpy(np.stack(actions, 0)).float(), dim=3, keepdim=True
        )
        # torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)

        # y_pred is the "chosen_action_qvals" in pymarl
        # y_target is the "target_max_qvals" in pymarl
        state = torch.mean(
            torch.from_numpy(np.stack(state, 0)).float(), 2, keepdim=True
        )

        # do stuff here like
        if self.mixer is not None:
            # inputs needs to include batch['state']
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


class COMATrainer(DQNTrainer):
    def __init__(
        self,
        qf,
        target_qf,
        policy=None,
        critic=None,
        target_critic=None,
        learning_rate=1e-3,
        soft_target_tau=1e-3,
        target_update_period=1,
        qf_criterion=None,
        discount=0.99,
        reward_scale=1.0,
    ):
        super().__init__(
            qf,
            target_qf,
            policy,
            None,
            None,
            learning_rate,
            soft_target_tau,
            target_update_period,
            qf_criterion,
            discount,
            reward_scale,
        )
        self.qf = qf
        if policy is None:
            self.policy = MAArgmaxDiscretePolicy(self.qf)
        else:
            self.policy = policy

        self.critic = critic
        self.target_critic = target_critic
        self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.critic_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.critic_params = list(self.critic.parameters())
        self.grad_norm_clip = 10  # copy from pymarl.config

    def calculate_qf(self, obs):
        """
        returns the output of the qf with RNN, or otherwise
        see here:
        https://github.com/ray-project/ray/blob/master/rllib/agents/qmix/model.py
        """
        if hasattr(self.qf, "init_hidden"):
            h = self.qf.init_hidden()
            agent_out = []
            max_seq_len = obs.shape[-1]
            for t in range(max_seq_len):
                # do stuff now.
                q, h = self.qf.forward(obs[:, t], h)
                agent_out.append(q)
            return torch.stack(agent_out, dim=1)
        else:
            return self.qf(obs_item)

    def _train_critic(self, obs, states, rewards, terminals, actions, active_agent):
        """
        we don't have avail actions in petting zoo envs?
        this is copied from coma_learner.py from pymarl
        """

        def build_td_lambda_targets(
            rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda
        ):
            # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
            # Initialise  last  lambda -return  for  not  terminated  episodes
            ret = target_qs.new_zeros(*target_qs.shape)
            # print("rewards", rewards.shape)  # coma only supports shared reward
            rewards = torch.max(rewards, -1)[1].unsqueeze(
                3
            )  # coma only supports shared reward
            terminated = terminated.permute(0, 1, 3, 2)
            mask = mask.permute(0, 1, 3, 2)
            ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
            # Backwards  recursive  update  of the "forward  view"
            for t in range(ret.shape[1] - 2, -1, -1):
                # print("ret", ret[:, t].shape)
                # print("mask", mask[:, t].shape)

                header = td_lambda * gamma * ret[:, t + 1]
                tail = rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (
                    1 - terminated[:, t]
                )
                # print(header.shape)
                # print(tail.shape)
                ret[:, t] = header + mask[:, t] * tail
            # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
            return ret[:, 0:-1]

        mask = active_agent
        bs = obs.shape[0]
        # Optimise critic
        # print("before obs tc", obs.shape)
        # print("before actions tc", actions.shape)
        target_q_vals = self.target_critic(
            obs, states, actions
        )  # this is an MLP, but with counterfactual inputs
        # print("after tc, target q val", target_q_vals.shape)
        # this "un-onehot"
        # torch.max(actions, -1)[1].unsqueeze(3).long()
        targets_taken = torch.gather(
            target_q_vals, dim=3, index=torch.max(actions, -1)[1].unsqueeze(3).long()
        )
        # print("target_q_vals", target_q_vals.shape)
        # print("targets_taken", targets_taken.shape)

        # Calculate td-lambda targets
        td_lambda = 0.8
        gamma = 0.99
        n_agents = actions.shape[
            -2
        ]  # not the best but leave for now - make sure things are padded
        n_actions = actions.shape[-1]
        targets = build_td_lambda_targets(
            rewards, terminals, active_agent, targets_taken, n_agents, gamma, td_lambda
        )

        # print("target_q_vals", target_q_vals.shape)

        q_vals = torch.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        # print("\n\n\tRunning Log\n\n")

        # across time, to calculate advantage-esque items
        # this is to determine the counter factual
        # we do it backwards from things that die.
        # print("mask", mask.shape)
        # print("mask[:, t]", mask[:, 1].shape)
        # print("rewards", rewards.shape)
        # print(reversed(range(rewards.size(1)-1)))
        for t in reversed(range(rewards.size(1) - 1)):
            mask_t = mask[:, t].squeeze(1).expand(-1, n_agents)
            # print("mask_t", mask_t.shape)
            if mask_t.sum() == 0:  # everyone is dead
                continue

            # todo this is an MLP, the second argument takes the t slice(?)
            # isn't it easier to compute all and take slices?
            # its probably because everything operates backwards, but we could still
            # do that conceptually...
            # otherwise this is fairly normal because modifying the input before it goes into the critic?
            q_t = self.critic(obs, states, actions, t)

            # print(q_vals.shape)
            # print(q_t.shape)

            q_vals[:, t] = q_t.squeeze(1)
            # torch.max(actions[:, t : t + 1], -1)[1].unsqueeze(3).long()
            q_taken = torch.gather(
                q_t,
                dim=3,
                index=torch.max(actions[:, t : t + 1], -1)[1].unsqueeze(3).long(),
            ).squeeze(1)
            targets_t = targets[:, t]
            # print("q_taken", q_taken.shape)
            # print("targets_t", targets_t.shape)

            td_error = (q_taken - targets_t.detach()).squeeze(2)

            # 0-out the targets that came from padded data
            # print("td_error", td_error.shape)
            # print("mask_t", mask_t.shape)
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_params, self.grad_norm_clip
            )
            self.critic_optimizer.step()
            # self.critic_training_steps += 1

            # stats for thing to track - we'll probalby pull this out and do it in the rlkit way.
            """
            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append(
                (masked_td_error.abs().sum().item() / mask_elems)
            )
            running_log["q_taken_mean"].append(
                (q_taken * mask_t).sum().item() / mask_elems
            )
            running_log["target_mean"].append(
                (targets_t * mask_t).sum().item() / mask_elems
            )
            """

        return q_vals, running_log

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        states = batch["states"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        active_agent = batch["active_agents"]

        # deal with ragged inputs later...
        obs = torch.from_numpy(np.stack(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.stack(terminals, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()
        rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
        states = torch.from_numpy(np.stack(states, axis=0)).float()
        active_agent = torch.from_numpy(np.stack(active_agent, axis=0)).float()

        # in the mixer setting they need to be managed in groups
        size = obs[0].shape[0]
        path_len = obs[0].shape[-1]
        batch_num = len(obs)

        # print("input-actions", actions.shape)

        # everything revolves around whole paths when using GRU

        """
        train critic here...
        """
        q_vals, critic_train_stats = self._train_critic(
            obs, states, rewards, terminals, actions, active_agent
        )
        q_vals = q_vals.detach()
        # print("critic trained!")
        # print("qvals", q_vals.shape)

        """
        Compute loss
        """
        # compute: best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
        # this is "equivalent" to self.qf(next_obs) and self.qf(obs)
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        batch_num = len(obs)
        obs_qs = []
        # print("batch_num", batch_num)
        for batch in range(batch_num):
            size = obs[batch].shape[1]
            path_len = obs[batch].shape[0]
            hidden = torch.cat(self.qf.init_hidden(size), 0)
            # best_action_idx = []
            obs_q = []
            for t in range(path_len - 1):
                q_ = []
                # for agent_indx in range(size):
                #    print(torch.from_numpy(obs[batch][t, agent_indx, :]).float().shape)
                #    print(hidden[agent_indx].float().shape)
                #    q, h = self.qf(torch.from_numpy(obs[batch][[t], agent_indx, :]).float(), hidden[agent_indx].float())
                #    hidden[agent_indx] = h
                #    q_.append(q)

                q, hidden = self.qf(
                    torch.from_numpy(obs[batch][t, :, :]).float(), hidden
                )
                obs_q.append(q)

            q, hidden = self.qf(
                torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden
            )
            obs_q = torch.stack(obs_q, 0)
            # if self.mixer is None:
            obs_qs.append(obs_q)
        obs_qs = torch.stack(obs_qs, 0)
        obs_qs = obs_qs / obs_qs.sum(dim=-1, keepdim=True)

        # Calculate baseline - be aware of the "off by one"
        # print("obs_qs", obs_qs.shape)
        # print("q_vals", q_vals.shape)
        baseline = (obs_qs * q_vals).sum(-1).detach()

        # TODO calculate policy grad with mask?
        q_taken = torch.gather(
            q_vals, dim=3, index=torch.max(actions[:, :-1], -1)[1].unsqueeze(3).long()
        ).squeeze(1)
        pi_taken = torch.gather(
            obs_qs, dim=3, index=torch.max(actions[:, :-1], -1)[1].unsqueeze(3).long()
        ).squeeze(1)
        active_agent = active_agent.permute(0, 1, 3, 2)
        pi_taken[active_agent[:, :-1] == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        # print("baseline", baseline.shape)
        # print("q_taken", q_taken.shape)
        advantages = q_taken.squeeze(3) - baseline
        # print("log_pi_taken", log_pi_taken.shape)
        coma_loss = -((advantages * log_pi_taken.squeeze(3))).sum()

        # Optimise agents
        self.qf_optimizer.zero_grad()
        coma_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.qf.parameters(), self.grad_norm_clip
        )
        self.qf_optimizer.step()

        """
        Update networks
        """
        # self.qf_optimizer.zero_grad()
        # qf_loss.backward()
        # self.qf_optimizer.step()

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)
            ptu.soft_update_from_to(
                self.critic, self.target_critic, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(coma_loss))
            # self.eval_statistics.update(
            #    create_stats_ordered_dict(
            #        "Y Predictions",
            #        ptu.get_numpy(y_pred),
            #    )
            # )

        self._n_train_steps_total += 1

    @property
    def networks(self):
        return [self.qf, self.target_qf, self.critic, self.target_critic]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
            critic=self.critic,
            target_critic=self.target_critic,
        )
