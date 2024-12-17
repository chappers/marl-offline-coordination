import numpy as np
import torch
from torch import nn as nn


from marlkit.policies.base import ExplorationPolicy, Policy
from marlkit.torch.core import eval_np
from marlkit.torch.distributions import TanhNormal
from marlkit.torch.networks import Mlp, RNNNetwork

import torch.nn.functional as F
import marlkit.torch.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)


class MLPPolicy(Mlp, ExplorationPolicy):
    """
    Usage - this is for discrete...

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim, init_w=1e-3, **kwargs):
        super().__init__(hidden_sizes, input_size=obs_dim, output_size=action_dim, init_w=init_w, **kwargs)
        self.action_dim = action_dim

    def get_log_proba(self, obs):
        # returns the log proba for munchausen RL
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        action_logits = self.last_fc(h)
        # action_probabilities = torch.clamp(action_logits, -32.0, 32.0)
        action_probabilities = torch.softmax(action_logits, -1)
        action_probabilities = torch.clamp(action_probabilities, 1e-8, 1)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = torch.distributions.Categorical(action_probabilities)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action = action_probabilities
        log_prob = torch.log(action + z)
        return log_prob

    def get_action(self, obs_np, deterministic=False):
        # print(len(obs_np))
        # print(obs_np[0].keys())
        if type(obs_np) is list:
            # check that it has a couple of keys
            actions = []
            for obs_dict in obs_np:
                actions.append(self.get_actions(obs_dict["obs"]))
            return np.array(actions).flatten(), {}
        else:
            actions = self.get_actions(obs_np[None], deterministic=deterministic)
            return actions[0], {}

    def get_actions(self, obs_np, deterministic=False):
        # print("obs_np", obs_np)
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions):
        # raw_actions = atanh(actions)
        # h = obs
        # for i, fc in enumerate(self.fcs):
        #     h = self.hidden_activation(fc(h))
        # action_logit = self.last_fc(h)

        # mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        # if self.std is None:
        #     log_std = self.last_fc_log_std(h)
        #     log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     std = torch.exp(log_std)
        # else:
        #     std = self.std
        #     log_std = self.log_std

        # tanh_normal = TanhNormal(mean, std)
        # log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        # return log_prob.sum(-1)

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        action_logits = self.last_fc(h)
        action_probabilities = torch.softmax(action_logits)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = torch.distributions.Categorical(action_probabilities)  # so that you can sample
        action = action_distribution.sample().cpu()

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action = action_probabilities
        log_prob = torch.log(action + z)
        return log_prob

    def forward(
        self,
        obs,
        reparameterize=True,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        action_logits = self.last_fc(h)
        # action_probabilities = torch.clamp(action_logits, -32.0, 32.0)
        action_probabilities = torch.softmax(action_logits, -1)
        action_probabilities = torch.clamp(action_probabilities, 1e-8, 1)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = torch.distributions.Categorical(action_probabilities)  # so that you can sample
        try:
            action = action_distribution.sample().cpu()
        except Exception as e:
            print(action_probabilities)
            print(e)
            raise Exception("")

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(action_probabilities + z)

        return action, action_probabilities, log_prob, action_logits
        # action, action_probabilities, log_action_probabilities, max_probability_action

        # log_prob = None
        # entropy = None
        # mean_action_log_prob = None
        # pre_tanh_value = None
        # if deterministic:
        #     action = torch.tanh(mean)
        # else:
        #     tanh_normal = TanhNormal(mean, std)
        #     if return_log_prob:
        #         if reparameterize is True:
        #             action, pre_tanh_value = tanh_normal.rsample(
        #                 return_pretanh_value=True
        #             )
        #         else:
        #             action, pre_tanh_value = tanh_normal.sample(
        #                 return_pretanh_value=True
        #             )
        #         log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
        #         log_prob = log_prob.sum(dim=1, keepdim=True)
        #     else:
        #         if reparameterize is True:
        #             action = tanh_normal.rsample()
        #         else:
        #             action = tanh_normal.sample()

        mean = None
        log_std = None
        entropy = None
        std = None
        mean_action_log_prob = None
        pre_tanh_value = None

        return (
            action,
            mean,
            log_std,
            log_prob,
            entropy,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )


class RNNPolicy(RNNNetwork, ExplorationPolicy):
    """
    Usage - this is for discrete...

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.

    ----

    It needs to match some of the signatures with RecurrentPolicy
    so that it performs rollouts properly
    """

    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super().__init__(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, **kwargs)
        self.action_dim = output_size
        self.hidden_states = None

    def init_hidden(self, size=None):
        # should be one item at a time...
        self.hidden_states = super().init_hidden(size)
        return self.hidden_states

    def reset(self):
        self.hidden_states = None

    def get_action_(self, obs, indx, deterministic=False):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        if indx is None:
            action, _, _, q, hidden = self.forward(obs, self.hidden_states, deterministic=deterministic)
        else:
            action, _, _, q, hidden = self.forward(obs, self.hidden_states[indx], deterministic=deterministic)
        action = ptu.get_numpy(action[0])
        return action, hidden

    def get_action(self, obs, deterministic=False):
        # we're in MARL land, with recurrency, see RecurrentPolicy.get_action
        if type(obs) is list:
            # in MARL its a list of dicts
            actions = []
            if self.hidden_states is None:
                self.init_hidden(size=len(obs))
            for indx, obs_dict in enumerate(obs):
                act, hidden = self.get_action_(obs_dict["obs"], indx, deterministic)
                self.hidden_states[indx] = hidden
                actions.append(act)
            return actions, {}
        else:
            act, hidden = self.get_action_(obs, deterministic=deterministic)
            self.hidden_states = hidden
            return act, {}

    def get_actions(self, obs_np, deterministic=False):
        # print("obs_np", obs_np)
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions):
        # raw_actions = atanh(actions)
        # h = obs
        # for i, fc in enumerate(self.fcs):
        #     h = self.hidden_activation(fc(h))
        # action_logit = self.last_fc(h)

        # mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        # if self.std is None:
        #     log_std = self.last_fc_log_std(h)
        #     log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     std = torch.exp(log_std)
        # else:
        #     std = self.std
        #     log_std = self.log_std

        # tanh_normal = TanhNormal(mean, std)
        # log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        # return log_prob.sum(-1)
        raise NotImplementedError
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        action_logits = self.last_fc(h)
        action_probabilities = torch.softmax(action_logits)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = torch.distributions.Categorical(
            action_probabilities
        )  # so that you can sample
        action = action_distribution.sample().cpu()

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        action = action_probabilities
        log_prob = torch.log(action + z)
        return log_prob
        """

    def forward(
        self,
        inputs,
        hidden_state,
        agent_indx=None,
        reparameterize=True,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if type(inputs) is list:  # for ease of use
            inputs = torch.cat(inputs, dim=1)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_sizes)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        action_probabilities = torch.softmax(q, -1)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = torch.distributions.Categorical(action_probabilities)  # so that you can sample
        action = action_distribution.sample().cpu()

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(action_probabilities + z)

        # action_logits = q
        return action, action_probabilities, log_prob, q, h


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(self, hidden_sizes, obs_dim, action_dim, std=None, init_w=1e-3, **kwargs):
        super().__init__(hidden_sizes, input_size=obs_dim, output_size=action_dim, init_w=init_w, **kwargs)
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions):
        raw_actions = atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
        self,
        obs,
        reparameterize=True,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
                else:
                    action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action,
            mean,
            log_std,
            log_prob,
            entropy,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation, deterministic=True)


class VAEPolicy(Mlp, ExplorationPolicy):
    def __init__(self, hidden_sizes, obs_dim, action_dim, latent_dim, std=None, init_w=1e-3, **kwargs):
        super().__init__(hidden_sizes, input_size=obs_dim, output_size=action_dim, init_w=init_w, **kwargs)
        self.latent_dim = latent_dim
        # working off a list is not implemented, have a look at Mlp helper
        # to understand how this would work in general
        hidden_size = int(np.mean(hidden_sizes))  # this is a hack

        self.e1 = torch.nn.Linear(obs_dim + action_dim, hidden_size)
        self.e2 = torch.nn.Linear(hidden_size, hidden_size)

        self.mean = torch.nn.Linear(hidden_size, self.latent_dim)
        self.log_std = torch.nn.Linear(hidden_size, self.latent_dim)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, hidden_size)
        self.d2 = torch.nn.Linear(hidden_size, hidden_size)
        self.d3 = torch.nn.Linear(hidden_size, action_dim)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic, execute_actions=True)[0]

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], -1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * ptu.from_numpy(np.random.normal(0, 1, size=(std.size())))

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(state.size(0), state.size(1), state.size(2), self.latent_dim))).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))

    # def decode_multiple(self, state, z=None, num_decode=10):
    #     if z is None:
    #         z = ptu.from_numpy(np.random.normal(0, 1, size=(state.size(0), num_decode, self.latent_dim))).clamp(
    #             -0.5, 0.5
    #         )

    #     a = F.relu(self.d1(torch.cat([state.unsqueeze(0).repeat(num_decode, 1, 1, 1, 1).permute(1, 0, 2), z], -1)))
    #     a = F.relu(self.d2(a))
    #     return torch.tanh(self.d3(a)), self.d3(a)
