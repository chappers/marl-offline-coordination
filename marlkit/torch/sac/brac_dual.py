"""
This implements BRAC? using the dual form KL divergence
The primal form is fairly trivial - just change MMD to KL divergence

**2021 02 05** - Gradient penalty is implemented.

TODO Value penalty variation rather than policy penalty - requires changing replay buffer to suport SARSA setup.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd


EPS = 1e-12


class BRACDualTrainer(TorchTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        discrim,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        plotter=None,
        render_eval_paths=False,
        # BEAR specific params
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
        self.discrim = discrim
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

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
        self.discrim_optimizer = optimizer_class(
            self.discrim.parameters(),
            lr=3e-4,
        )

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
        self.policy_update_style = policy_update_style
        self.target_mmd_thresh = target_mmd_thresh

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.discrete = False
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

    def eval_q_custom(self, custom_policy, data_batch, q_function=None):
        if q_function is None:
            q_function = self.qf1

        obs = data_batch["observations"]
        # Evaluate policy Loss
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )
        q_new_actions = q_function(obs, new_obs_actions)
        return float(q_new_actions.mean().detach().cpu().numpy())

    def kl_div_estimate(self, logits_p, logits_b):
        return -nn.Softplus()(logits_b) + torch.log(nn.Softplus()(logits_p) + EPS) + 1.0

    def kl_div_loss(self, obs, actor_actions, actions, estimate, gamma=5.0):
        batch_size = actions.size()[0]

        # calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(actions)
        if ptu.gpu_enabled():
            alpha = alpha.cuda()

        interpolated = alpha * actions.data + (1 - alpha) * actor_actions.data
        interpolated = autograd.Variable(interpolated, requires_grad=True)
        prob_interpolated = self.discrim(obs, interpolated)
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda()
            if ptu.gpu_enabled()
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + EPS)
        gradient_penalty = gamma * ((gradients_norm - 1) ** 2).mean()
        return -estimate.mean() + gradient_penalty

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        Learn the discriminator in the dual form
        """
        # in the dual form, we train the discriminator function
        actor_samples, _, _, _, _, _, _, raw_actor_actions = self.policy(
            obs,
            return_log_prob=True,
        )
        logits_discrim_pol = self.discrim(obs, raw_actor_actions)
        logits_discrim_beh = self.discrim(obs, actions)

        discrim_estimate = self.kl_div_estimate(logits_discrim_pol, logits_discrim_beh)
        # BRAC does some gradient penalty
        discrim_loss = self.kl_div_loss(obs, raw_actor_actions, actions, discrim_estimate)
        self.discrim_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)  # this is to keep discrim_estimate
        self.discrim_optimizer.step()

        # recon, mean, std = self.vae(obs, actions)
        # recon_loss = self.qf_criterion(recon, actions)
        # kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # vae_loss = recon_loss + 0.5 * kl_loss

        # self.vae_optimizer.zero_grad()
        # vae_loss.backward()
        # self.vae_optimizer.step()

        """
        Critic Training
        """
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            # Compute value of perturbed actions sampled
            action_rep = self.policy(next_obs)[0]
            target_qf1 = self.target_qf1(next_obs, action_rep)
            target_qf2 = self.target_qf2(next_obs, action_rep)

            # Soft Clipped Double Q-learning
            target_Q = 0.75 * torch.min(target_qf1, target_qf2) + 0.25 * torch.max(target_qf1, target_qf2)
            target_Q = target_Q.view(next_obs.shape[0], -1).max(1)[0].view(-1, 1)

            # for the value penalty, we additionally regularise - the best way is to enable SARSA in the
            # replay buffer so we can do it in a sane way - otherwise we can skip it will look something like
            # target_Q = (
            #     self.reward_scale * rewards
            #     + (1.0 - terminals) * self.discount * (target_Q - alpha * next_discrim_estimate)
            # )
            target_Q = self.reward_scale * rewards + (1.0 - terminals) * self.discount * target_Q

        qf1_pred = self.qf1(obs, actions)
        qf2_pred = self.qf2(obs, actions)

        qf1_loss = (qf1_pred - target_Q.detach()).pow(2).mean()
        qf2_loss = (qf2_pred - target_Q.detach()).pow(2).mean()

        """
        Actor Training
        """
        # sampled_actions, raw_sampled_actions = self.vae.decode_multiple(
        #     obs, num_decode=self.num_samples_mmd_match
        # )
        # the dual form does not use vae, it uses a discriminator function

        actor_samples, _, _, _, _, _, _, raw_actor_actions = self.policy(
            obs,
            return_log_prob=True,
        )

        if self.kernel_choice == "kl":
            # not required, just use
            # discrim_estimate
            beh_penalty = discrim_estimate
        elif self.kernel_choice == "laplacian":
            raise Exception("Not implemented for BRAC Dual")
            # beh_penalty = self.mmd_loss_laplacian(
            #    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma
            # )
        elif self.kernel_choice == "gaussian":
            raise Exception("Not implemented for BRAC Dual")
            # beh_penalty = self.mmd_loss_gaussian(
            #    raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma
            # )

        action_divergence = ((actions - actor_samples) ** 2).sum(-1)
        raw_action_divergence = ((actions - raw_actor_actions) ** 2).sum(-1)

        q_val1 = self.qf1(obs, actor_samples[:, :])
        q_val2 = self.qf2(obs, actor_samples[:, :])

        if self.policy_update_style == "0":
            policy_loss = torch.min(q_val1, q_val2)[:, 0]
        elif self.policy_update_style == "1":
            policy_loss = torch.mean(q_val1, q_val2)[:, 0]

        if self._n_train_steps_total >= 40000:
            # Now we can update the policy
            if self.mode == "auto":
                policy_loss = (-policy_loss + self.log_alpha.exp() * (beh_penalty - self.target_mmd_thresh)).mean()
            else:
                policy_loss = (-policy_loss + 100 * beh_penalty).mean()
        else:
            if self.mode == "auto":
                policy_loss = (self.log_alpha.exp() * (beh_penalty - self.target_mmd_thresh)).mean()
            else:
                policy_loss = 100 * beh_penalty.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self.policy_optimizer.zero_grad()
        if self.mode == "auto":
            policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        if self.mode == "auto":
            self.alpha_optimizer.zero_grad()
            (-policy_loss).backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=-5.0, max=10.0)

        """
        Update networks
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """
        Some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Num Q Updates"] = self._num_q_update_steps
            self.eval_statistics["Num Policy Updates"] = self._num_policy_update_steps
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(qf1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(qf2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q Targets",
                    ptu.get_numpy(target_Q),
                )
            )
            self.eval_statistics.update(create_stats_ordered_dict("MMD Loss", ptu.get_numpy(beh_penalty)))
            self.eval_statistics.update(
                create_stats_ordered_dict("Action Divergence", ptu.get_numpy(action_divergence))
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Raw Action Divergence", ptu.get_numpy(raw_action_divergence))
            )
            if self.mode == "auto":
                self.eval_statistics["Alpha"] = self.log_alpha.exp().item()

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
            self.discrim,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            discrim=self.discrim,
        )
