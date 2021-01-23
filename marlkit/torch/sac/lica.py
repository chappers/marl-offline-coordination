import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.torch_rl_algorithm import MATorchTrainer


class GumbelSoftmax(OneHotCategorical):
    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -torch.log(-torch.log(U + self.eps))

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return torch.softmax(y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()


def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy()


class LICACritic(nn.Module):
    def __init__(self, n_actions, n_agents, state_shape, mixing_embed_dim):
        super(LICACritic, self).__init__()

        self.n_actions = n_actions
        self.n_agents = n_agents
        self.state_shape = state_shape

        self.output_type = "q"

        # Set up network layers
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, self.hid_dim), nn.ReLU(), nn.Linear(self.hid_dim, self.hid_dim)
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim), nn.ReLU(), nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        bs = states.size(0)
        states = states.reshape(-1, self.state_dim)
        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions)

        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)
        b1 = b1.view(-1, 1, self.hid_dim)

        h = torch.relu(torch.bmm(action_probs, w1) + b1)

        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.hid_dim, 1)

        h2 = torch.bmm(h, w_final)

        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        q = h2 + b2

        q = q.view(bs, -1, 1)

        return q


class LICATrainer(MATorchTrainer):
    # based on this: https://github.com/mzho7212/LICA/blob/main/src/learners/lica_learner.py
    def __init__(
        self,
        env,
        policy,
        critic,
        target_critic,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        critic_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        plotter=None,
        render_eval_paths=False,
        use_automatic_entropy_tuning=False,
        target_entropy=0.11,  # see lica.yml
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.critic = critic
        self.target_critic = target_critic
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            raise NotImplementedError
            """
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
            """
        else:
            self.target_entropy = target_entropy  # this is called entropy_coef in LICA

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.critic_optimizer = optimizer_class(self.critic.parameters(), lr=critic_lr)

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
        states = batch["states"]

        """
        Policy and Alpha Loss
        """
        # no need to worry about groups of games. in the IAC setting.
        obs = torch.from_numpy(np.concatenate(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.concatenate(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.concatenate(terminals, axis=0)).float()
        actions = torch.from_numpy(np.concatenate(actions, axis=0)).float()
        rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).float()
        states = torch.from_numpy(np.concatenate(states, axis=0)).float()

        # see https://github.com/mzho7212/LICA/blob/main/src/run.py
        # for the runner
        # self.train_critic_td

        # optimise critic
        target_q_vals = self.target_critic(actions, states)  # check dim, and reformat - it should be one hot

        # calculate td-lambda targets
        targets = rewards + (1.0 - terminals) * self.discount * target_q_vals
        q_t = self.critic(actions[:, :-1], states[:, :-1])
        td_error = q_t - targets.detach()  # ensure right size

        # normal l2 loss, over mean of data
        # we don't have mask in these envs
        critic_loss = (td_error ** 2).mean()

        # self.train
        # policy out is the logits w/o softmax
        agent_outs = self.policy(obs)
        mac_out_entropy = multinomial_entropy(agent_outs).mean(dim=-1, keepdim=True)
        mac_out = torch.nn.functional.softmax(agent_outs, dim=-1)

        # mix action proba and state to estimate joint q-value
        if self.critic is not None:
            mix_loss = self.critic(mac_out, states)

            # - not sure if needed as we don't use mask in non-SMAC envs
            # mask = mask.expand_as(mix_loss)
            # entropy_mask = copy.deepcopy(mask)

            # mix_loss = (mix_loss * mask).sum() / mask.sum()

            # Adaptive Entropy Regularization
            # entropy_loss = (mac_out_entropy * entropy_mask).sum() / entropy_mask.sum()
            entropy_loss = mac_out_entropy.mean()
            entropy_ratio = self.target_entropy / entropy_loss.item()

            mix_loss = -mix_loss - entropy_ratio * entropy_loss
        else:
            raise Exception("This doesn't work without a mixer?")

        """
        Update networks
        """
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm_clip = 10
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_norm_clip)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        mix_loss.backward()
        grad_norm_clip = 10  # copied from the config settings
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_norm_clip)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.critic, self.target_critic, self.soft_target_tau)

        """
        Save some statistics for eval
        use similar to LICA
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics["Critic Loss"] = np.mean(ptu.get_numpy(critic_loss))
            self.eval_statistics["Critic Grad Norm"] = np.mean(ptu.get_numpy(critic_grad_norm))
            self.eval_statistics["Target Mean"] = np.mean(ptu.get_numpy(targets))
            self.eval_statistics["TD Error Abs"] = np.mean(np.abs(ptu.get_numpy(td_error)))
            self.eval_statistics["Q_T mean"] = np.mean(ptu.get_numpy(q_t))

            # policy stats
            self.eval_statistics["Mix Loss"] = np.mean(ptu.get_numpy(mix_loss))
            self.eval_statistics["Entropy Loss"] = np.mean(ptu.get_numpy(entropy_loss))
            self.eval_statistics["Agent Grad Norm"] = np.mean(ptu.get_numpy(grad_norm))
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
            self.critic,
            self.target_critic,
        ]

    def get_snapshot(self):
        return dict(policy=self.policy, critic=self.critic, target_critic=self.target_critic)
