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

from sklearn.preprocessing import StandardScaler, MinMaxScaler



class DBRTrainer(MATorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        vae_pos,
        vae_neg,
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
        
        # obts settings similar to BEAR
        mode="auto",
        kernel_choice="laplacian",
        policy_update_style=0,
        mmd_sigma=10.0,
        target_mmd_thresh=0.05,
        num_samples_mmd_match=4,
        use_target_nets=True,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vae_neg = vae_neg
        self.vae_pos = vae_pos
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

        # for behaviour training
        self.vae_neg_optimizer = optimizer_class(
            self.vae_neg.parameters(),
            lr=3e-4,
        )
        self.vae_pos_optimizer = optimizer_class(
            self.vae_pos.parameters(),
            lr=3e-4,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        
        # obts/bear settings
        self.mode = mode
        if self.mode == "auto":
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=1e-3,
            )
        self.mmd_sigma = mmd_sigma
        self.kernel_choice = kernel_choice
        self.num_samples_mmd_match = num_samples_mmd_match
        self.policy_update_style = policy_update_style  # not used, assumed to be 0
        self.target_mmd_thresh = target_mmd_thresh

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        # # once bitten twice shy - split batch via mean
        self.standard_scalar = StandardScaler()
        self.minmax_scalar = MinMaxScaler()
        
    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean(
            (-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2)
        )

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean(
            (-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2)
        )

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean(
            (-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2)
        )

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean(
            (-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2)
        )

        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        diff_x_y = torch.mean(
            (-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2)
        )

        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean(
            (-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2)
        )

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def multi_mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Laplacian kernel for support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        action_dim = samples1.shape[-1]
        # print("shape", samples1.shape)
        return torch.stack(
            [
                self.mmd_loss_laplacian(
                    samples1[:, :, :, idx].unsqueeze(-1),
                    samples2[:, :, :, idx].unsqueeze(-1),
                    sigma,
                )
                for idx in range(action_dim)
            ],
            -1,
        )

    def multi_mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """MMD constraint with Gaussian Kernel support matching"""
        # sigma is set to 20.0 for hopper, cheetah and 50 for walker/ant
        action_dim = samples1.shape[-1]
        return torch.stack(
            [
                self.mmd_loss_gaussian(
                    samples1[:, :, idx].unsqueeze(2),
                    samples2[:, :, idx].unsqueeze(2),
                    sigma,
                )
                for idx in range(action_dim)
            ],
            -1,
        )

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
        total_qf1_loss = []
        total_qf2_loss = []
        total_q1_preds = []
        total_q2_preds = []
        total_q_target = []
        total_policy_loss = []

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

            rewards = rewards.unsqueeze(0)
            terminals = terminals.unsqueeze(0)
            obs = obs.unsqueeze(0)
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            active_agent = active_agent.unsqueeze(0)
            # state_0 = batch["states_0"]
            actions = actions.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)

            """
            Split datasets for beheaviour-style learning. Keep the original still.
            its part of: `self.standard_scalar.mean_`
            There maybe something here to take the polyak average by doing a 
            `scalar.fit.weights * alpha + scalar.weights or similar`
            """
            # print(rewards.detach().numpy().shape)
            # print(rewards.detach().numpy().mean(axis=(0, 2, 3)).shape)
            # print(rewards.mean(axis=(-1, -2)).shape)
            self.standard_scalar.partial_fit(rewards.mean(axis=(0, 2, 3)).detach().numpy().reshape(-1, 1))
            self.minmax_scalar.partial_fit(rewards.mean(axis=(0, 2, 3)).detach().numpy().reshape(-1, 1))
            neg_indx_standard = rewards.mean(axis=(0, 2, 3)) <= float(self.standard_scalar.mean_)
            neg_indx_minmax = rewards.mean(axis=(0, 2, 3)) <= (
                (float(self.minmax_scalar.data_max_) + float(self.minmax_scalar.data_min_))
                / 2
            )

            # print(np.mean(neg_indx_minmax.detach().numpy()))
            # print(np.mean(neg_indx_standard.detach().numpy()))
            # print((float(self.minmax_scalar.data_max_) + float(self.minmax_scalar.data_min_))/2)

            # pick the one which is closer to half of the samples...
            if np.abs(np.mean(neg_indx_minmax.detach().numpy()) - 0.5) < np.abs(
                np.mean(neg_indx_standard.detach().numpy()) - 0.5
            ):
                # print("minmax")
                neg_indx = neg_indx_minmax
            else:
                # print("standard")
                neg_indx = neg_indx_standard
            pos_indx = ~neg_indx
            

            # rewards_neg = rewards.index_select(0, neg_indx.nonzero().long().view(-1))
            # terminals_neg = terminals.index_select(0, neg_indx.nonzero().long().view(-1))
            obs_neg = obs.index_select(1, neg_indx.nonzero().long().view(-1))
            actions_neg = actions.index_select(1, neg_indx.nonzero().long().view(-1))
            # next_obs_neg = next_obs.index_select(0, neg_indx.nonzero().long().view(-1))

            # rewards_pos = rewards.index_select(0, (pos_indx).nonzero().long().view(-1))
            # terminals_pos = terminals.index_select(0, (pos_indx).nonzero().long().view(-1))
            obs_pos = obs.index_select(1, (pos_indx).nonzero().long().view(-1))
            actions_pos = actions.index_select(1, (pos_indx).nonzero().long().view(-1))
            # next_obs_pos = next_obs.index_select(0, (pos_indx).nonzero().long().view(-1))
            

            """
            Behavior clone the policies...
            # print("rewards", rewards.shape)
            # print("rewards neg", rewards_neg.shape)
            # print("rewards pos", rewards_pos.shape)
            # print("actions", actions.shape)
            # print("actions neg", actions_neg.shape)
            # print("actions pos", actions_pos.shape)
            # print("obs", obs.shape)
            # print("obs neg", obs_neg.shape)
            # print("obs pos", obs_pos.shape)
            # print("\n")
            """

            # vae_pos, vae_neg is optimized with the filters items
            recon_neg, mean_neg, std_neg = self.vae_neg(obs_neg, actions_neg)
            recon_neg_loss = self.qf_criterion(recon_neg, actions_neg)
            kl_loss_neg = (
                -0.5
                * (1 + torch.log(std_neg.pow(2)) - mean_neg.pow(2) - std_neg.pow(2)).mean()
            )
            vae_loss_neg = recon_neg_loss + 0.5 * kl_loss_neg

            self.vae_neg_optimizer.zero_grad()
            vae_loss_neg.backward()
            self.vae_neg_optimizer.step()

            recon_pos, mean_pos, std_pos = self.vae_pos(obs_pos, actions_pos)
            recon_pos_loss = self.qf_criterion(recon_pos, actions_pos)
            kl_loss_pos = (
                -0.5
                * (1 + torch.log(std_pos.pow(2)) - mean_pos.pow(2) - std_pos.pow(2)).mean()
            )
            vae_loss_pos = recon_pos_loss + 0.5 * kl_loss_pos

            self.vae_pos_optimizer.zero_grad()
            vae_loss_pos.backward()
            self.vae_pos_optimizer.step()


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
            _, action_probs, log_pis, new_obs_raw_actions = self.policy(
                obs,
                reparameterize=True,
                return_log_prob=True,
            )
                
            # add behavior actions comparison - we'll use KL divergence
            # we'll sample off all obs but then filter with the mask afterwards
            # so that we can add the two items together as they'll be the same shape
            raw_sampled_actions_neg = self.vae_neg.decode(obs)
            raw_sampled_actions_pos = self.vae_pos.decode(obs)

            # we'll compare the distance of the sampled and raw in proba...
            # compare: `raw_sampled_actions_neg + raw_sampled_actions_pos`
            # as per BRAC it is in the form KL(pi, behavior_pi)
            # new_obs_raw_actions = new_obs_raw_actions.repeat(
            #     1, self.num_samples_mmd_match, 1
            # ).view(obs.shape[0], self.num_samples_mmd_match, actions.shape[1])
            if self.kernel_choice == "laplacian":
                mmd_neg_loss = self.multi_mmd_loss_laplacian(
                    raw_sampled_actions_neg, new_obs_raw_actions, sigma=self.mmd_sigma
                )
                mmd_pos_loss = self.multi_mmd_loss_laplacian(
                    raw_sampled_actions_pos, new_obs_raw_actions, sigma=self.mmd_sigma
                )
                mmd_behavior_pen = torch.tanh(
                    self.multi_mmd_loss_laplacian(
                        raw_sampled_actions_pos,
                        raw_sampled_actions_neg,
                        sigma=self.mmd_sigma,
                    )
                )
            elif self.kernel_choice == "gaussian":
                mmd_neg_loss = self.multi_mmd_loss_gaussian(
                    raw_sampled_actions_neg, new_obs_raw_actions, sigma=self.mmd_sigma
                )
                mmd_pos_loss = self.multi_mmd_loss_gaussian(
                    raw_sampled_actions_pos, new_obs_raw_actions, sigma=self.mmd_sigma
                )
                mmd_behavior_pen = torch.tanh(
                    self.multi_mmd_loss_gaussian(
                        raw_sampled_actions_pos,
                        raw_sampled_actions_neg,
                        sigma=self.mmd_sigma,
                    )
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
                
                    
            # assume policy_update_style == "0" not using self.policy_update_style
            q1_new_actions = self.qf1(torch.cat([obs, action_probs], -1))
            q1_new_actions = q1_new_actions.detach()
            q2_new_actions = self.qf2(torch.cat([obs, action_probs], -1))
            q2_new_actions = q2_new_actions.detach()
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)

            if self.mode == "auto":
                mmd_loss = (
                    self.log_alpha.exp() * mmd_behavior_pen * (mmd_neg_loss - mmd_pos_loss)
                )
            else:
                mmd_loss = 100 * mmd_behavior_pen * (mmd_neg_loss - mmd_pos_loss)

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
                q1_new_actions = q1_new_actions.detach()
                q2_new_actions = self.qf2(torch.cat([obs, action_probs], -1))
                q2_new_actions = q2_new_actions.detach()
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
                policy_loss = (mmd_loss + policy_loss).mean()
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

                policy_loss = (mmd_loss + action_probs * (alpha * log_pis - q_new_actions)).mean()

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
