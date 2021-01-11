import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, qs, *args):
        """Forward pass for the mixer.
        Args:
            agent_qs: Tensor of shape [B, T, n_agents, n_actions]
            states: Tensor of shape [B, T, state_dim]
        """
        return torch.sum(qs, dim=2, keepdim=True)


class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        # rllib style
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        """Forward pass for the mixer.
        Args:
            agent_qs: Tensor of shape [B, T, n_agents, n_actions]
            states: Tensor of shape [B, T, state_dim]
        """
        bs = agent_qs.size(0)
        n_agents = agent_qs.size(2)
        states = states.reshape(-1, self.state_dim)
        # try unsafe/dynamic mode
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        # unsafe/dynamic mode
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


class QTranBase(nn.Module):
    def __init__(self, n_agents, n_actions, state_shape, mixing_embed_dim, rnn_hidden_dim):
        super(QTranBase, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = mixing_embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions

        # Q(s,u)
        q_input_size = self.state_dim + self.rnn_hidden_dim + self.n_actions

        self.Q = nn.Sequential(
            nn.Linear(q_input_size, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        # V(s)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
        ae_input = self.rnn_hidden_dim + self.n_actions
        self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input), nn.ReLU(), nn.Linear(ae_input, ae_input))

    def forward(self, obs, actions, states, hidden_states):
        agent_state_action_input = torch.cat([hidden_states, actions], dim=2)
        agent_state_action_encoding = self.action_encoding(agent_state_action_input)
        agent_state_action_encoding = agent_state_action_encoding.sum(dim=1)  # Sum across agents

        inputs = torch.cat([states.squeeze(1), agent_state_action_encoding], dim=1)
        q_outputs = self.Q(inputs)
        v_outputs = self.V(states)

        return q_outputs, v_outputs
