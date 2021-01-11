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


class QCGraphQmix(nn.Module):
    # QMIX variation
    def __init__(self, n_agents, state_shape, mixing_embed_dim, rnn_hidden_dim):
        super(QCGraphQmix, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.rnn_hidden_dim = rnn_hidden_dim

        self.embed_dim = mixing_embed_dim

        # cluster assignment
        self.assign_agent = nn.Linear(self.rnn_hidden_dim, self.n_agents)

        # state bias?
        self.assign_bias = nn.Sequential(nn.Linear(self.state_dim, 1), nn.Softmax(-1))

        # gcn filter
        self.graph_w1 = nn.Linear(self.rnn_hidden_dim, self.embed_dim)

        # qmix style embedding info
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # qmix style v(s)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states, hidden_states):
        # assign each agent to another agent
        agent_score = self.assign_agent(hidden_states)
        bs = agent_score.shape[0]
        state_score = self.assign_bias(states).repeat(1, self.n_agents, self.n_agents)

        embedding_eye = (
            (torch.eye(self.n_agents, self.n_agents).unsqueeze(2))
            .expand(self.n_agents, self.n_agents, bs)
            .permute(2, 0, 1)
        )
        agent_score = agent_score + (torch.eye(self.n_agents, self.n_agents).unsqueeze(0) * -1e16)
        oneness_knn = F.gumbel_softmax(agent_score, hard=True)
        dynamic_knn = agent_score.softmax(-1) > state_score

        # this provides agents with a hard assignment to at least one other agent
        cluster_output = embedding_eye + dynamic_knn + oneness_knn
        cluster_output = torch.clamp(cluster_output, 0, 1)

        # normalizing rowwise as per kipf et al.
        adj_d = torch.sqrt(torch.sum(cluster_output, dim=-2, keepdim=True) * embedding_eye)
        cluster_output = torch.bmm(adj_d, cluster_output)
        cluster_output = torch.bmm(cluster_output, adj_d)

        # add gcn - this is guarenteed to be monotonic
        w1 = torch.bmm(cluster_output, agent_qs.permute(0, 2, 1))

        # follow qmix, make sure monotonic
        # first layer
        # print(torch.bmm(agent_qs, w1).shape) # 1 32!
        # w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)

        # w1 is size
        hidden = F.elu(w1 + b1)

        # second layer
        w_final = self.hyper_w_final(states).permute(0, 2, 1)

        # apply learned diffpool weights to "unpool"
        # this is also monotonic
        AF = torch.bmm(cluster_output, hidden)
        s = torch.bmm(AF, w_final)
        # State-dependent bias
        v_outputs = self.V(states)

        # now w1 needs to be pooled via graph pooling
        q_tot = torch.bmm(hidden, w_final) + v_outputs
        return q_tot


class QCGraph(nn.Module):
    def __init__(self, n_agents, n_actions, state_shape, mixing_embed_dim, rnn_hidden_dim):
        super(QCGraph, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))
        self.embed_dim = mixing_embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions

        # cluster assignment
        self.assign_agent = nn.Linear(self.rnn_hidden_dim + n_actions, self.n_agents)

        # state bias?
        self.assign_bias = nn.Sequential(nn.Linear(self.state_dim, 1), nn.Softmax(-1))

        # gcn filter
        self.graph_w1 = nn.Linear(self.rnn_hidden_dim + self.n_actions, self.embed_dim)

        # qmix style embedding info
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # qmix style v(s)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, obs, actions, states, hidden_states):
        # copy the signature of QTran for easy integration
        agent_state_action_input = torch.cat([hidden_states, actions], dim=2)  # input for our graph

        # assign each agent to another agent
        agent_score = self.assign_agent(agent_state_action_input)  # B, n_agent, n_agent
        bs = agent_score.shape[0]
        state_score = self.assign_bias(states).repeat(1, self.n_agents, self.n_agents)  # B, n_agent, n_agent

        embedding_eye = (
            (torch.eye(self.n_agents, self.n_agents).unsqueeze(2))
            .expand(self.n_agents, self.n_agents, bs)
            .permute(2, 0, 1)
        )
        agent_score = agent_score + (torch.eye(self.n_agents, self.n_agents).unsqueeze(0) * -1e16)
        oneness_knn = F.gumbel_softmax(agent_score, hard=True)
        dynamic_knn = agent_score.softmax(-1) > state_score

        # this provides agents with a hard assignment to at least one other agent
        cluster_output = embedding_eye + dynamic_knn + oneness_knn
        cluster_output = torch.clamp(cluster_output, 0, 1)

        # normalizing rowwise as per kipf et al.
        adj_d = torch.sqrt(torch.sum(cluster_output, dim=-2, keepdim=True) * embedding_eye)
        cluster_output = torch.bmm(adj_d, cluster_output)
        cluster_output = torch.bmm(cluster_output, adj_d)

        # add gcn
        agent_all_perm_approx = torch.bmm(cluster_output, agent_state_action_input)

        # now follow qmix - first layer
        w1 = self.graph_w1(agent_all_perm_approx)
        b1 = self.hyper_b_1(states)
        hidden = F.elu(w1 + b1)
        # print(hidden.shape) # B, nagent, embedding

        # second layer (i.e. diff pool)
        w_final = self.hyper_w_final(states).permute(0, 2, 1)

        # apply learned diffpool weights to "unpool"
        AF = torch.bmm(cluster_output, hidden)
        s = torch.bmm(AF, w_final)
        # State-dependent bias
        v_outputs = self.V(states)

        # compute final output
        q_output = torch.bmm(hidden, w_final) + v_outputs
        q_output_pooled = torch.bmm(q_output.permute(0, 2, 1), s)
        return q_output_pooled, v_outputs
