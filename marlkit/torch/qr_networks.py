"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from marlkit.policies.base import Policy
from marlkit.torch import pytorch_util as ptu
from marlkit.torch.core import eval_np, np_ify
from marlkit.torch.data_management.normalizer import TorchFixedNormalizer
from marlkit.torch.modules import LayerNorm

import numpy as np


def identity(x):
    return x


class LinearTransform(nn.Module):
    def __init__(self, m, b):
        super().__init__()
        self.m = m
        self.b = b

    def __call__(self, t):
        return self.m * t + self.b


class QRMlp(nn.Module):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        num_quant=200,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.action_size = output_size  # this is the action space...
        self.output_size = output_size * num_quant
        self.num_quant = num_quant
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, self.output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False, return_action=False):
        if len(input.shape) == 3:
            bs = input.size(0)
        else:
            bs = None
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        # if bs is not None:
        #     print(bs, output.shape, input.shape)
        if bs is not None:
            output = output.view(bs, -1, self.action_size, self.num_quant)
        else:
            output = output.view(-1, self.action_size, self.num_quant)
        # print(output.shape)
        if return_preactivations:
            return output, preactivation
        elif return_action:
            # print(input.shape, output.shape)
            # print(output.mean(2).squeeze(0))
            if bs is not None:
                return output.mean(3).squeeze(0)
            else:
                return output.mean(2).squeeze(0)
        else:
            return output


class QRMlpPolicy(QRMlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(self, *args, obs_normalizer: TorchFixedNormalizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action_(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        actions = self.forward(obs, return_action=True)
        return np_ify(actions).argmax()

    def get_action(self, obs):
        if type(obs) is list:
            # we're in MARL land
            actions = []
            for obs_dict in obs:
                actions.append(self.get_action_(obs_dict["obs"]))
            return actions, {}
        else:
            obs = np.expand_dims(obs, axis=0)
            obs = ptu.from_numpy(obs).float()
            actions = self.forward(obs, return_action=True)
            return np_ify(actions).argmax(), {}

    def get_actions(self, obs):
        return eval_np(self, obs, return_action=True)
