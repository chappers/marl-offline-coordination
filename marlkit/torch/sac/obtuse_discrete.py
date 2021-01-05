from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Normal

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.torch_rl_algorithm import TorchTrainer

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class OBTTrainer(TorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        vae_neg,
        vae_pos,
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

        # in AWAC/MRL
        self.lo = -1  # copying from the M-SAC implementation
        self.m_alpha = 0.9  # copying from the M-SAC implementation
        self.m_tau = 0.03  # copying from the M-SAC implementation
        self.mask_positive_advantage = True

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env.action_space.shape
                ).item()  # heuristic value from Tuomas
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

        # for behaviour learning
        self.vae_neg_optimizer = optimizer_class(
            self.vae_neg.parameters(),
            lr=3e-4,
        )
        self.vae_pos_optimizer = optimizer_class(
            self.vae_pos.parameters(),
            lr=3e-4,
        )

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

        # once bitten twice shy - split batch via mean
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
        return torch.stack(
            [
                self.mmd_loss_laplacian(
                    samples1[:, :, idx].unsqueeze(2),
                    samples2[:, :, idx].unsqueeze(2),
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
        # need to learn the mid point so that we
        # split the datasets in the appropriate groups...
        # this can be done without grad...
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Make use of the advantage as per AWAC style.
        """
        awac_style = True
        if awac_style:
            with torch.no_grad():
                new_obs_actions, policy_mle, _, log_pi, *_ = self.policy(
                    obs,
                    reparameterize=True,
                    return_log_prob=True,
                )

                # Advantage Weighted Regression to filter
                # assume awr_use_mle_for_vf is false, and vf_K == 1
                # awr_sample_actions = false, buffer_policy_sample_actions = false
                # awr_min_q = false, use_automatic_beta_tuning=false
                # normalize_over_state -> advantage
                # self.mask_positive_advantage -> true(?)
                v1_pi = self.qf1(obs)
                v2_pi = self.qf2(obs)
                v_pi = torch.min(v1_pi, v2_pi)
                u = actions
                q_adv = self.qf1(next_obs)
                score = q_adv - v_pi
                score = torch.max(score, 1)[0]
                # if self.mask_positive_advantage:  # this is to split the obs for further processing.
                pos_indx = score > 0
                neg_indx = score <= 0

                rewards_neg = rewards.index_select(
                    0, neg_indx.nonzero().long().view(-1)
                )
                terminals_neg = terminals.index_select(
                    0, neg_indx.nonzero().long().view(-1)
                )
                obs_neg = obs.index_select(0, neg_indx.nonzero().long().view(-1))
                actions_neg = actions.index_select(
                    0, neg_indx.nonzero().long().view(-1)
                )
                next_obs_neg = next_obs.index_select(
                    0, neg_indx.nonzero().long().view(-1)
                )

                rewards_pos = rewards.index_select(
                    0, (pos_indx).nonzero().long().view(-1)
                )
                terminals_pos = terminals.index_select(
                    0, (pos_indx).nonzero().long().view(-1)
                )
                obs_pos = obs.index_select(0, (pos_indx).nonzero().long().view(-1))
                actions_pos = actions.index_select(
                    0, (pos_indx).nonzero().long().view(-1)
                )
                next_obs_pos = next_obs.index_select(
                    0, (pos_indx).nonzero().long().view(-1)
                )
        else:
            self.standard_scalar.partial_fit(rewards.detach().numpy())
            self.minmax_scalar.partial_fit(rewards.detach().numpy())
            neg_indx_standard = rewards.reshape(-1) <= float(self.standard_scalar.mean_)
            neg_indx_minmax = rewards.reshape(-1) <= (
                (
                    float(self.minmax_scalar.data_max_)
                    + float(self.minmax_scalar.data_min_)
                )
                / 2
            )

            # pick the one which is closer to half of the samples...
            if np.abs(np.mean(neg_indx_minmax.detach().numpy()) - 0.5) < np.abs(
                np.mean(neg_indx_standard.detach().numpy()) - 0.5
            ):
                neg_indx = neg_indx_minmax
            else:
                neg_indx = neg_indx_standard
            pos_indx = ~neg_indx

            rewards_neg = rewards.index_select(0, neg_indx.nonzero().long().view(-1))
            terminals_neg = terminals.index_select(
                0, neg_indx.nonzero().long().view(-1)
            )
            obs_neg = obs.index_select(0, neg_indx.nonzero().long().view(-1))
            actions_neg = actions.index_select(0, neg_indx.nonzero().long().view(-1))
            next_obs_neg = next_obs.index_select(0, neg_indx.nonzero().long().view(-1))

            rewards_pos = rewards.index_select(0, (pos_indx).nonzero().long().view(-1))
            terminals_pos = terminals.index_select(
                0, (pos_indx).nonzero().long().view(-1)
            )
            obs_pos = obs.index_select(0, (pos_indx).nonzero().long().view(-1))
            actions_pos = actions.index_select(0, (pos_indx).nonzero().long().view(-1))
            next_obs_pos = next_obs.index_select(
                0, (pos_indx).nonzero().long().view(-1)
            )

        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

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

        """
        Policy and Alpha Loss
        """
        """
        (
            new_obs_actions,
            policy_mean,
            policy_log_std,
            log_pi,
            _,
            _,
            _,
            new_obs_raw_actions,
        ) = self.policy(obs, reparameterize=True, return_log_prob=True,)
        """
        _, action_prob, log_pi, new_obs_raw_actions = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        # add behavior actions comparison - we'll use KL divergence
        # we'll sample off all obs but then filter with the mask afterwards
        # so that we can add the two items together as they'll be the same shape
        sampled_actions_neg, raw_sampled_actions_neg = self.vae_neg.decode_multiple(obs)
        sampled_actions_pos, raw_sampled_actions_pos = self.vae_pos.decode_multiple(obs)

        # we'll compare the distance of the sampled and raw in proba...
        # compare: `raw_sampled_actions_neg + raw_sampled_actions_pos`
        # as per BRAC it is in the form KL(pi, behavior_pi)
        new_obs_raw_actions = new_obs_raw_actions.repeat(
            1, self.num_samples_mmd_match, 1
        ).view(obs.shape[0], self.num_samples_mmd_match, actions.shape[1])
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

        # calculate stats...
        # skip for now

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        # assume policy_update_style == "0" not using self.policy_update_style
        q_new_actions = torch.min(
            self.qf1(obs),
            self.qf2(obs),
        )

        if self.mode == "auto":
            mmd_loss = (
                self.log_alpha.exp() * mmd_behavior_pen * (mmd_neg_loss - mmd_pos_loss)
            )
        else:
            mmd_loss = 100 * mmd_behavior_pen * (mmd_neg_loss - mmd_pos_loss)

        policy_loss = (mmd_loss + action_prob * (alpha * log_pi - q_new_actions)).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs)
        q2_pred = self.qf2(obs)

        # Make sure policy accounts for squashing functions like tanh correctly!
        _, new_action_prob, new_log_pi, _ = self.policy(
            next_obs,
            reparameterize=True,
            return_log_prob=True,
        )
        target_q_values = (
            new_action_prob
            * torch.min(
                self.target_qf1(next_obs),
                self.target_qf2(next_obs),
            )
            - alpha * new_log_pi
        )

        """
        Munchausen Variation
        """

        mean_neg_sample = raw_sampled_actions_neg.mean(1)
        mean_pos_sample = raw_sampled_actions_pos.mean(1)

        std_neg_sample = raw_sampled_actions_neg.std(1)
        std_pos_sample = raw_sampled_actions_pos.std(1)

        dist_pos = Normal(mean_pos_sample, std_pos_sample)  # .reshape(m_shape)
        dist_neg = Normal(mean_neg_sample, std_neg_sample)  # .reshape(m_shape)

        log_pi_a = self.m_tau * (
            dist_pos.log_prob(actions).mean(1).unsqueeze(1).cpu()
            - dist_neg.log_prob(actions).mean(1).unsqueeze(1).cpu()
        )

        munchausen_reward = self.reward_scale * rewards + self.m_alpha * torch.clamp(
            log_pi_a, min=self.lo, max=0
        )
        q_target = (
            munchausen_reward + (1.0 - terminals) * self.discount * target_q_values
        )
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
            policy_loss = (log_pi - q_new_actions).mean()

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
