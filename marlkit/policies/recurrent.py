"""
recurrent policy for handling the GRU units
"""

import numpy as np
from torch import nn

import marlkit.torch.pytorch_util as ptu
from marlkit.policies.base import Policy


class RecurrentPolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf
        self.hidden_states = None

    def reset(self):
        self.hidden_states = None

    def init_hidden(self, size=None):
        # should be one item at a time...
        self.hidden_states = self.qf.init_hidden(size)

    def get_action_(self, obs, agent_indx=None):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        if agent_indx is None:
            q_values, hidden = self.qf(obs, self.hidden_states)
        else:
            q_values, hidden = self.qf(obs, self.hidden_states[agent_indx])
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), hidden

    """
    def get_action(self, obs):
        if type(obs) is list:
            # we're in MARL land...
            actions = []
            for obs_dict in obs:
                actions.append(self.get_action_(obs_dict["obs"])[0])
            return actions, {}
        else:
            return self.get_action_(obs)
    """

    def get_action(self, obs):
        if type(obs) is list:
            # we're in MARL land...
            actions = []
            if self.hidden_states is None:
                self.init_hidden(size=len(obs))
            for indx, obs_dict in enumerate(obs):
                act, hidden = self.get_action_(obs_dict["obs"], indx)
                self.hidden_states[indx] = hidden
                actions.append(act)
            return actions, {}
        else:
            act, hidden = self.get_action_(obs)
            self.hidden_states = hidden
            return act, {}
