"""
An implementation of Mixer variation just to familiarise myself with using this custom setup. 
We'll assume the usage of the RNN agent as well, so we'll need to handle the hidden states here
"""

import numpy as np
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer


class DoubleDQNTrainer(DQNTrainer):
    def calculate_qf(self, obs):
        """
        returns the output of the qf with RNN, or otherwise
        see here:
        https://github.com/ray-project/ray/blob/master/rllib/agents/qmix/model.py
        """
        if hasattr(self.qf, "init_hidden"):
            h = self.qf.init_hidden()
            agent_out = []
            max_seq_len = obs.shape[-1]
            for t in range(max_seq_len):
                # do stuff now.
                q, h = self.qf.forward(obs[:, t], h)
                agent_out.append(q)
            return torch.stack(agent_out, dim=1)
        else:
            return self.qf(obs_item)

    def train_from_torch(self, batch):
        print(batch.keys())
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        state = batch["states"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # in the mixer setting they need to be managed in groups
        size = obs[0].shape[0]
        path_len = obs[0].shape[-1]
        batch_num = len(obs)

        # everything revolves around whole paths when using GRU

        """
        Compute loss
        """
        # compute: best_action_idxs = self.qf(next_obs).max(1, keepdim=True)[1]
        # this is "equivalent" to self.qf(next_obs) and self.qf(obs)
        best_action_idxs = []
        obs_qs = []
        for batch in range(batch_num):
            size = obs[batch].shape[1]
            path_len = obs[batch].shape[0]
            hidden = torch.cat(self.qf.init_hidden(size), 0)
            best_action_idx = []
            obs_q = []
            for t in range(path_len):
                q_ = []
                # for agent_indx in range(size):
                #    print(torch.from_numpy(obs[batch][t, agent_indx, :]).float().shape)
                #    print(hidden[agent_indx].float().shape)
                #    q, h = self.qf(torch.from_numpy(obs[batch][[t], agent_indx, :]).float(), hidden[agent_indx].float())
                #    hidden[agent_indx] = h
                #    q_.append(q)

                q, hidden = self.qf(
                    torch.from_numpy(obs[batch][t, :, :]).float(), hidden
                )
                if t != 0:
                    best_action_idx.append(q)
                obs_q.append(q)

            q, hidden = self.qf(
                torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden
            )
            best_action_idx.append(q)
            best_action_idx = torch.stack(best_action_idx, 0)
            obs_q = torch.stack(obs_q, 0)
            # if self.mixer is None:
            if True:
                best_action_idx = best_action_idx.max(-1, keepdim=True)[1]
            best_action_idxs.append(best_action_idx.permute(0, 2, 1))
            obs_qs.append(obs_q)

        best_action_idxs = torch.stack(best_action_idxs, 0)
        obs_qs = torch.stack(obs_qs, 0)

        # compute: target_q_values = self.target_qf(next_obs).gather(1, best_action_idxs).detach()
        target_q_values = []
        with torch.no_grad():
            for batch in range(batch_num):
                size = obs[batch].shape[1]
                path_len = obs[batch].shape[0]
                hidden = torch.cat(self.target_qf.init_hidden(size), 0)
                target_q_value = []
                for t in range(path_len):
                    q_ = []
                    q, hidden = self.target_qf(
                        torch.from_numpy(obs[batch][t, :, :]).float(), hidden
                    )
                    if t != 0:
                        target_q_value.append(q)

                q, hidden = self.qf(
                    torch.from_numpy(next_obs[batch][-1, :, :]).float(), hidden
                )
                target_q_value.append(q)
                target_q_value = torch.stack(target_q_value, 0)
                # if self.mixer is None:
                if True:
                    target_q_value = target_q_value.max(-1, keepdim=True)[1]
                target_q_values.append(target_q_value.permute(0, 2, 1))

        print("obs_qs", obs_qs.shape)

        # need to gather..., by best_action_idxs
        target_q_values = torch.stack(target_q_values, 0)
        y_target = (
            torch.from_numpy(np.stack(rewards, axis=0)).float()
            + (1.0 - torch.from_numpy(np.stack(terminals, axis=0)).float())
            * self.discount
            * target_q_values
        )
        y_target = y_target.detach()
        y_target = y_target.permute(0, 1, 3, 2)

        # actions is a one-hot vector
        y_pred = torch.sum(
            obs_qs * torch.from_numpy(np.stack(actions, 0)).float(), dim=3, keepdim=True
        )
        # torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)

        # y_pred is the "chosen_action_qvals" in pymarl
        # y_target is the "target_max_qvals" in pymarl
        print("state", np.stack(state, 0).shape)
        state = torch.mean(
            torch.from_numpy(np.stack(state, 0)).float(), 2, keepdim=True
        )
        print("state", state.shape)
        print("y_pred", y_pred.shape)
        print("y_target", y_target.shape)

        # do stuff here like
        if self.mixer is not None:
            # inputs needs to include batch['state']
            y_pred = self.mixer(y_pred, state)
            y_target = self.target_mixer(y_target, state).detach()

        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Y Predictions",
                    ptu.get_numpy(y_pred),
                )
            )

        self._n_train_steps_total += 1
