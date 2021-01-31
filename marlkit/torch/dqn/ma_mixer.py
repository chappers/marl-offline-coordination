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

            best_action_idxs = self.qf(next_obs).max(-1, keepdim=True)[1]
            # print(best_action_idxs.shape)
            # print(self.target_qf(next_obs).shape)
            target_q_values = self.target_qf(next_obs).gather(-1, best_action_idxs).detach()
            target_q_values = target_q_values.permute(0, 2, 1)

            if self.mrl:
                mrl_log_proba = torch.log(self.qf(obs).softmax(-1))

                mrl_log_proba = torch.max(mrl_log_proba, -1, keepdim=True)[0].permute(0, 2, 1)
                mrl_log_proba = mrl_log_proba[:, :, torch.randperm(mrl_log_proba.size(-1))]
                # print(mrl_log_proba.shape, rewards.shape)
                rewards = rewards + mrl_log_proba

            # print(target_q_values.shape)
            y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
            y_target = y_target.detach()
            y_target_mixer = None
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

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

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
                        qf_loss = torch.exp(log_pi - log_pi[:, :, [ag], :]).detach() * qf_loss_[:, :, [ag], :]
                    else:
                        qf_loss += torch.exp(log_pi - log_pi[:, :, [ag], :]).detach() * qf_loss_[:, :, [ag], :]

                qf_loss = qf_loss.mean()
            else:
                y_target = y_target.permute(0, 2, 1)
                if y_target_mixer is not None:
                    y_target_mixer = y_target_mixer.permute(0, 2, 1)
                    qf_mixer_loss = self.qf_criterion(y_pred, y_target_mixer)
                else:
                    qf_mixer_loss = None

                if self.inverse_weight:
                    # this should give equal weighting?
                    qf_base_loss = self.qf_criterion(y_pred, y_target)
                    mixer_weight = 1 / qf_mixer_loss
                    base_weight = 1 / qf_base_loss
                    total_weight = mixer_weight + base_weight
                    qf_loss = (mixer_weight / total_weight) * qf_mixer_loss + (
                        base_weight / total_weight
                    ) * qf_base_loss
                elif qf_mixer_loss is not None:
                    qf_loss = qf_mixer_loss
                else:
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
            return self.qf(obs)

    def _train_critic(self, obs, states, rewards, terminals, actions, active_agent):
        """
        we don't have avail actions in petting zoo envs?
        this is copied from coma_learner.py from pymarl
        """

        def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
            # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
            # Initialise  last  lambda -return  for  not  terminated  episodes
            ret = target_qs.new_zeros(*target_qs.shape)
            # print("rewards", rewards.shape)  # coma only supports shared reward
            rewards = torch.max(rewards, -1)[1].unsqueeze(3)  # coma only supports shared reward
            terminated = terminated.permute(0, 1, 3, 2)
            mask = mask.permute(0, 1, 3, 2)
            ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
            # Backwards  recursive  update  of the "forward  view"
            for t in range(ret.shape[1] - 2, -1, -1):
                # print("ret", ret[:, t].shape)
                # print("mask", mask[:, t].shape)

                header = td_lambda * gamma * ret[:, t + 1]
                tail = rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t])
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
        target_q_vals = self.target_critic(obs, states, actions)  # this is an MLP, but with counterfactual inputs
        # print("after tc, target q val", target_q_vals.shape)
        # this "un-onehot"
        # torch.max(actions, -1)[1].unsqueeze(3).long()
        targets_taken = torch.gather(target_q_vals, dim=3, index=torch.max(actions, -1)[1].unsqueeze(3).long())
        # print("target_q_vals", target_q_vals.shape)
        # print("targets_taken", targets_taken.shape)

        # Calculate td-lambda targets
        td_lambda = 0.8
        gamma = 0.99
        n_agents = actions.shape[-2]  # not the best but leave for now - make sure things are padded
        n_actions = actions.shape[-1]
        targets = build_td_lambda_targets(rewards, terminals, active_agent, targets_taken, n_agents, gamma, td_lambda)

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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_norm_clip)
            self.critic_optimizer.step()
            # self.critic_training_steps += 1

            # stats for thing to track - we'll probalby pull this out and do it in the rlkit way.
        return q_vals, running_log

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        states = batch["states"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        active_agent = batch["active_agents"]

        """
        # deal with ragged inputs later...
        obs = torch.from_numpy(np.stack(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.stack(terminals, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()
        rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
        states = torch.from_numpy(np.stack(states, axis=0)).float()
        active_agent = torch.from_numpy(np.stack(active_agent, axis=0)).float()
        """

        def to_tensor(x):
            try:
                return torch.from_numpy(np.array(x, dtype=float)).float()
            except:
                x = np.stack([np.array(x_, dtype=float).flatten()[np.newaxis, :] for x_ in x], 0)
                return torch.from_numpy(x).float()

        total_coma_loss = []

        for b in range(len(obs)):
            rewards = to_tensor(batch["rewards"][b])
            terminals = to_tensor(batch["terminals"][b])
            obs = to_tensor(batch["observations"][b])
            states = to_tensor(batch["states"][b])
            active_agent = to_tensor(batch["active_agents"][b])
            # state_0 = batch["states_0"]
            actions = to_tensor(batch["actions"][b])
            next_obs = to_tensor(batch["next_observations"][b])

            rewards = rewards.unsqueeze(0)
            terminals = terminals.unsqueeze(0)
            obs = obs.unsqueeze(0)
            states = states.unsqueeze(0)
            active_agent = active_agent.unsqueeze(0)
            actions = actions.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)

            # in the mixer setting they need to be managed in groups
            size = obs[0].shape[0]
            path_len = obs[0].shape[-1]
            batch_num = len(obs)

            # print("input-actions", actions.shape)

            # everything revolves around whole paths when using GRU

            """
            train critic here...
            """
            q_vals, critic_train_stats = self._train_critic(obs, states, rewards, terminals, actions, active_agent)
            q_vals = q_vals.detach()
            # print("critic trained!")
            # print("qvals", q_vals.shape)

            """
            Compute loss
            """
            # compute: best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
            # this is "equivalent" to self.qf(next_obs) and self.qf(obs)
            obs = [batch["observations"][b]]
            next_obs = [batch["next_observations"][b]]
            batch_num = len(obs)
            obs = torch.from_numpy(np.stack(obs, axis=0)).float()
            obs_qs = self.qf(obs[:, :-1])
            obs_qs = obs_qs / obs_qs.sum(dim=-1, keepdim=True)

            # Calculate baseline - be aware of the "off by one"
            # print("obs_qs", obs_qs.shape)
            # print("q_vals", q_vals.shape)
            baseline = (obs_qs * q_vals).sum(-1).detach()

            # TODO calculate policy grad with mask?
            q_taken = torch.gather(q_vals, dim=3, index=torch.max(actions[:, :-1], -1)[1].unsqueeze(3).long()).squeeze(
                1
            )
            pi_taken = torch.gather(obs_qs, dim=3, index=torch.max(actions[:, :-1], -1)[1].unsqueeze(3).long()).squeeze(
                1
            )
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
            grad_norm = torch.nn.utils.clip_grad_norm_(self.qf.parameters(), self.grad_norm_clip)
            self.qf_optimizer.step()

            """
            Update networks
            """
            # self.qf_optimizer.zero_grad()
            # qf_loss.backward()
            # self.qf_optimizer.step()
            total_coma_loss.append(ptu.get_numpy(coma_loss))

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)
            ptu.soft_update_from_to(self.critic, self.target_critic, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(total_coma_loss)
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
