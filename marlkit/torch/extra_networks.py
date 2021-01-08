import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module):
    def __init__(self, n_agents, action_size, obs_shape, state_shape, mixing_embed_dim):
        super(COMACritic, self).__init__()

        self.n_actions = action_size
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_size = action_size
        self.input_shape = (
            state_shape + obs_shape + action_size * n_agents * 2 + n_agents
        )
        self.mixing_embed_dim = mixing_embed_dim

        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, self.mixing_embed_dim)
        self.fc2 = nn.Linear(self.mixing_embed_dim, self.mixing_embed_dim)
        self.fc3 = nn.Linear(self.mixing_embed_dim, self.n_actions)

    def forward(self, obs, states, actions, t=None):
        inputs = self._build_inputs(obs, states, actions, t=t)
        # print("fwd, inputs", inputs.shape)
        # print("calc, inputs", self.input_shape)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, obs, states, actions, t=None):
        # bs = batch.batch_size
        # max_t = batch.max_seq_length if t is None else 1
        # we'll just force it so it runs and fix it later..
        # print(obs.shape)
        bs = obs.shape[0]
        max_t = obs.shape[1]  # guess - overwrite later...
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        # print("state", states.shape)
        # print("state", states[:, ts].shape)
        inputs.append(states[:, ts].repeat(1, 1, self.n_agents, 1))

        # observation
        # print(ts)
        inputs.append(obs[:, ts])
        # print("inputs", inputs[0].shape, inputs[1].shape)
        max_t = inputs[0].shape[1]

        # actions (masked out by agent)
        # print("actions[:, ts]", actions[:, ts].shape)
        # print("actions[:, ts].view", actions[:, ts].view(bs, max_t, 1, -1).shape)
        actions_expand = (
            actions[:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        )
        # actions_expand = actions[:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = 1 - th.eye(self.n_agents)
        agent_mask = (
            (agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # print("action", actions.shape)
        # print("agent_mask", agent_mask.shape)
        inputs.append(actions_expand * agent_mask)
        # print(
        #     "inputs",
        #     th.cat(
        #         [x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1
        #     ).shape,
        # )
        # self.obs_shape = obs_shape
        # self.state_shape = state_shape
        # self.action_size
        # print(
        #     "test", self.obs_shape + self.state_shape + self.action_size * self.n_agents
        # )
        # print(
        #     "test2",
        #     self.obs_shape
        #     + self.state_shape
        #     + self.action_size * self.n_agents
        #     + self.action_size * self.n_agents,
        # )

        # last actions

        if t == 0:
            last_actions = (
                th.zeros_like(actions[:, 0:1])
                .view(bs, max_t, 1, -1)
                .repeat(1, 1, self.n_agents, 1)
            )
        elif isinstance(t, int):
            last_actions = (
                actions[:, slice(t - 1, t)]
                .view(bs, max_t, 1, -1)
                .repeat(1, 1, self.n_agents, 1)
            )
        else:
            last_actions = th.cat(
                [th.zeros_like(actions[:, 0:1]), actions[:, :-1]], dim=1
            )
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(
                1, 1, self.n_agents, 1
            )
        # print("actions", actions.shape)
        # print("last_actions", last_actions.shape)
        inputs.append(last_actions)  # this should be n_agents * action_dim

        # self.eye
        self_reference = (
            th.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)
        )
        # print("self eye", self_reference.shape)
        inputs.append(self_reference)

        inputs = th.cat(
            [x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1
        )
        return inputs

    def _get_input_shape(self, scheme):
        # NOT USED
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # agent id
        input_shape += self.n_agents
        return input_shape
