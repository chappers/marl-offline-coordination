"""
recurrent policy for handling the GRU units
"""

import numpy as np
from torch import nn

import marlkit.torch.pytorch_util as ptu
from marlkit.policies.base import Policy
from torch.nn import functional as F
from torch.distributions import Categorical


class RecurrentPolicy(nn.Module, Policy):
    def __init__(self, qf, use_gumbel_softmax=False, eval_policy=False):
        super().__init__()
        self.qf = qf
        self.hidden_states = None
        self.use_gumbel_softmax = use_gumbel_softmax
        self.eval_policy = eval_policy

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
        #
        if self.use_gumbel_softmax and self.eval_policy:
            act_proba = F.gumbel_softmax(q_values, hard=True)
            act = Categorical(act_proba).sample().long()
            act_np = ptu.get_numpy(act)
            return act_np, hidden
        elif self.use_gumbel_softmax and not self.eval_policy:
            act_proba = F.gumbel_softmax(q_values, hard=False)
            # sample from here?
            act = Categorical(act_proba).sample().long()
            act_np = ptu.get_numpy(act)
            return act_np, hidden
        else:
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
                actions.append(act[0])
            return actions, {}
        else:
            act, hidden = self.get_action_(obs)
            self.hidden_states = hidden
            print("r actions", act)
            return act, {}
