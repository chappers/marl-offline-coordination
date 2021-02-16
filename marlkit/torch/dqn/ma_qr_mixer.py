"""
An implementation of IQL/Mixers using Quantile Regression 
This is for new paper.

Also implements conservative variation.

Double Conservative Q Learning for Multi-Agent Reinforcement Learning
"""

import numpy as np
import torch

import marlkit.torch.pytorch_util as ptu
from marlkit.core.eval_util import create_stats_ordered_dict
from marlkit.torch.dqn.ma_dqn import DQNTrainer
from marlkit.policies.argmax import MAArgmaxDiscretePolicy
import torch.optim as optim
from collections import OrderedDict


class CQLTrainer(DQNTrainer):
    num_quant = 200
    min_q_weight = 1.0

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        state = batch["states"]
        active_agent = batch["active_agents"]
        # state_0 = batch["states_0"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        # no need to worry about groups of games. in the IQL setting.
        # to do support batch esp. as games are different lengths...
        """
        obs = torch.from_numpy(np.stack(obs, axis=0)).float()
        next_obs = torch.from_numpy(np.stack(next_obs, axis=0)).float()
        terminals = torch.from_numpy(np.stack(terminals, axis=0)).float()
        actions = torch.from_numpy(np.stack(actions, axis=0)).float()
        rewards = torch.from_numpy(np.stack(rewards, axis=0)).float()
        active_agent = torch.from_numpy(np.stack(active_agent, axis=0)).float()
        """

        def to_tensor(x, filter_n=None):
            try:
                if filter_n is None:
                    return torch.from_numpy(np.array(x, dtype=float)).float()
                else:
                    return torch.from_numpy(np.array(x[:filter_n], dtype=float)).float()
            except:
                x = [np.array(x_) for x_ in x]
                if filter_n is None:
                    x = np.stack(x, 0)
                else:
                    x = x[:filter_n]
                    x = np.stack(x, 0)
                return torch.from_numpy(x).float()

        total_qf_loss = []
        total_y_pred = []

        self.num_quant = self.qf.num_quant  # hack

        for b in range(len(obs)):
            try:
                rewards = to_tensor(batch["rewards"][b])
                terminals = to_tensor(batch["terminals"][b])
                obs = to_tensor(batch["observations"][b])
                state = to_tensor(batch["states"][b])
                active_agent = to_tensor(batch["active_agents"][b])
                # state_0 = batch["states_0"]
                actions = to_tensor(batch["actions"][b])
                next_obs = to_tensor(batch["next_observations"][b])
                next_states = to_tensor(batch["next_states"][b])
            except:
                filter_n = len(batch["observations"][b]) - 1
                rewards = to_tensor(batch["rewards"][b], filter_n)
                terminals = to_tensor(batch["terminals"][b], filter_n)
                obs = to_tensor(batch["observations"][b], filter_n)
                state = to_tensor(batch["states"][b], filter_n)
                active_agent = to_tensor(batch["active_agents"][b], filter_n)
                # state_0 = batch["states_0"]
                actions = to_tensor(batch["actions"][b], filter_n)
                next_obs = to_tensor(batch["next_observations"][b], filter_n)
                next_states = to_tensor(batch["next_states"][b], filter_n)

            """
            Compute loss
            """
            # rewards = rewards.reshape(-1, 1).float()
            # terminals = terminals.reshape(-1, 1).float()
            # print(next_obs.shape)
            action_next = self.qf(next_obs, return_action=True)
            # print(action_next.shape)
            best_action_idx = action_next.max(-1, keepdim=True)[1]
            # print(best_action_idx.shape)
            best_action_idx = torch.nn.functional.one_hot(best_action_idx, self.qf.action_size)

            if len(best_action_idx.shape) == 3:
                best_action_idx = best_action_idx.unsqueeze(0)
            best_action_idx = best_action_idx.permute(0, 1, 3, 2)

            target_q_all = self.target_qf(next_obs).detach()
            target_q_values = target_q_all.gather(1, best_action_idx.repeat(1, 1, 1, self.num_quant))
            # target_q_values = target_q_values.permute(0, 2, 1)

            # print(target_q_values.shape, terminals.shape, rewards.shape)
            terminals = terminals.permute(0, 2, 1).unsqueeze(-1)
            rewards = rewards.permute(0, 2, 1).unsqueeze(-1)
            # print(target_q_values.shape, terminals.shape, rewards.shape)
            y_target = rewards + (1.0 - terminals) * self.discount * target_q_values
            y_target_mixer = None
            # actions is a one-hot vector
            y_pred = self.qf(obs) * actions.unsqueeze(-1)
            # print(y_pred.shape)

            state = torch.from_numpy(np.stack(state, 0)).float()
            qf_loss = None

            if self.mixer is None:
                y_target_mixer = None
                bellman_errors = y_target - y_pred  # for QR
                # eqn 9
                kappa = 1.0  # set this properly please
                huber_loss = torch.where(
                    bellman_errors.abs() < kappa,
                    0.5 * bellman_errors.pow(2),
                    kappa * (bellman_errors.abs() - 0.5 * kappa),
                )
                # quantile midpoints - lemma 2
                tau_hat = torch.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)
                qf_loss = huber_loss * ((tau_hat - bellman_errors.detach() < 0).float().abs())
                qf_loss = qf_loss.mean()

                if self.cql:
                    # add conservative q learning loss here
                    replay_chosen_q = self.qf(obs) * actions.unsqueeze(-1)
                    dataset_expec = replay_chosen_q.mean()
                    negative_sampling = torch.logsumexp(replay_chosen_q, dim=1).mean()

                    min_q_loss = (negative_sampling - dataset_expec) * self.min_q_weight

                    if self.with_lagrange:
                        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                        min_q_loss = alpha_prime * (min_q_loss - self.target_action_gap)

                        self.alpha_prime_optimizer.zero_grad()
                        alpha_prime_loss = (-min_q_loss) * 0.5
                        alpha_prime_loss.backward(retain_graph=True)
                        self.alpha_prime_optimizer.step()

                    qf_loss = qf_loss + min_q_loss

            elif self.mixer is not None and not self.cql_double:
                # apply mixer to calculate qf loss.
                # inputs needs to include batch['state']

                y_pred = y_pred.mean(3).permute(0, 2, 1)
                y_target = y_target.mean(3).permute(0, 2, 1)
                # print(y_pred.shape, y_target.shape)

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

                # apply loss
                bellman_errors = y_target_mixer - y_pred  # for QR
                # eqn 9
                kappa = 1.0  # set this properly please
                huber_loss = torch.where(
                    bellman_errors.abs() < kappa,
                    0.5 * bellman_errors.pow(2),
                    kappa * (bellman_errors.abs() - 0.5 * kappa),
                )
                # quantile midpoints - lemma 2
                tau_hat = torch.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)
                qf_loss = huber_loss * ((tau_hat - bellman_errors.detach() < 0).float().abs())
                qf_loss = qf_loss.mean()

                if self.cql:
                    # add conservative q learning loss here
                    replay_chosen_q = self.qf(obs) * actions.unsqueeze(-1)
                    replay_chosen_q = replay_chosen_q.mean(3).permute(0, 2, 1)
                    replay_chosen_q = self.mixer(replay_chosen_q, state)
                    dataset_expec = replay_chosen_q.mean()
                    negative_sampling = torch.logsumexp(replay_chosen_q, dim=1).mean()

                    min_q_loss = (negative_sampling - dataset_expec) * self.min_q_weight

                    if self.with_lagrange:
                        alpha_prime_mixer = torch.clamp(self.log_alpha_prime_mixer.exp(), min=0.0, max=1000000.0)
                        min_q_loss = alpha_prime_mixer * (min_q_loss - self.target_action_gap)

                        self.alpha_prime_mixer_optimizer.zero_grad()
                        alpha_prime_mixer_loss = (-min_q_loss) * 0.5
                        alpha_prime_mixer_loss.backward(retain_graph=True)
                        self.alpha_prime_mixer_optimizer.step()

                    qf_loss = qf_loss + min_q_loss
            elif self.mixer is not None and self.cql_double and self.cql:
                y_target_mixer = None
                bellman_errors = y_target - y_pred  # for QR
                # eqn 9
                kappa = 1.0  # set this properly please
                huber_loss = torch.where(
                    bellman_errors.abs() < kappa,
                    0.5 * bellman_errors.pow(2),
                    kappa * (bellman_errors.abs() - 0.5 * kappa),
                )
                # quantile midpoints - lemma 2
                tau_hat = torch.Tensor((2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)).view(1, -1)
                qf_loss = huber_loss * ((tau_hat - bellman_errors.detach() < 0).float().abs())
                qf_loss = qf_loss.mean()

                # add conservative q learning loss here
                replay_chosen_q = self.qf(obs) * actions.unsqueeze(-1)
                dataset_expec = replay_chosen_q.mean(dim=2)
                negative_sampling = torch.logsumexp(replay_chosen_q, dim=2)  # .mean()
                # print(replay_chosen_q.shape, negative_sampling.shape)

                min_q_loss = (negative_sampling - dataset_expec) * self.min_q_weight
                ### qf_loss = qf_loss + min_q_loss.mean()

                ###########################
                ## now do the mixer part ##
                ###########################
                y_pred = y_pred.mean(3).permute(0, 2, 1)
                y_target = y_target.mean(3).permute(0, 2, 1)
                # print(y_pred.shape, y_target.shape)

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

                y_pred = self.mixer(y_pred, state)
                y_target_mixer = self.target_mixer(y_target, state).detach()

                # apply loss
                bellman_errors_mixer = y_target_mixer - y_pred  # for QR
                # eqn 9
                kappa = 1.0  # set this properly please
                huber_loss_mixer = torch.where(
                    bellman_errors_mixer.abs() < kappa,
                    0.5 * bellman_errors_mixer.pow(2),
                    kappa * (bellman_errors_mixer.abs() - 0.5 * kappa),
                )
                # quantile midpoints - lemma 2
                # tau_hat = torch.Tensor(
                #    (2 * np.arange(self.num_quant) + 1) / (2.0 * self.num_quant)
                # ).view(1, -1)
                qf_loss_mixer = huber_loss_mixer * ((tau_hat - bellman_errors_mixer.detach() < 0).float().abs())
                qf_loss_mixer = qf_loss_mixer.mean()

                # add conservative q learning loss here
                replay_chosen_q_mixer = self.qf(obs) * actions.unsqueeze(-1)
                replay_chosen_q_mixer = replay_chosen_q_mixer.mean(3).permute(0, 2, 1)
                replay_chosen_q_mixer = self.mixer(replay_chosen_q_mixer, state)
                dataset_expec_mixer = replay_chosen_q_mixer.mean(2)
                negative_sampling_mixer = torch.logsumexp(replay_chosen_q_mixer, dim=2)  # .mean()

                min_q_loss_mixer = (negative_sampling_mixer - dataset_expec_mixer) * self.min_q_weight
                ### qf_loss_mixer = qf_loss_mixer + min_q_loss_mixer.mean()

                # now time to combine loss together, so that we can penality correctly
                # for the greedy/non-greedy actions

                if self.with_lagrange:
                    alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                    min_q_loss = alpha_prime * (min_q_loss - self.target_action_gap)

                    self.alpha_prime_optimizer.zero_grad()
                    alpha_prime_loss = (-min_q_loss.mean()) * 0.5
                    alpha_prime_loss.backward(retain_graph=True)
                    self.alpha_prime_optimizer.step()

                    alpha_prime_mixer = torch.clamp(self.log_alpha_prime_mixer.exp(), min=0.0, max=1000000.0)
                    min_q_loss_mixer = alpha_prime_mixer * (min_q_loss_mixer - self.target_action_gap)

                    self.alpha_prime_mixer_optimizer.zero_grad()
                    alpha_prime_mixer_loss = (-min_q_loss_mixer.mean()) * 0.5
                    alpha_prime_mixer_loss.backward(retain_graph=True)
                    self.alpha_prime_mixer_optimizer.step()

                    # the weighting on each one is inverse to the alpha...
                    # this seems kind of ironic
                    # basically if alpha is close to 0, it should have higher weight than the alternative.
                    if self.inverse_weight:
                        mixer_weight = (1 / (alpha_prime_mixer + 1e-12)).detach()
                        indep_weight = (1 / (alpha_prime + 1e-12)).detach()
                        total_weight = mixer_weight + indep_weight
                        q_loss_both = (indep_weight / total_weight) * min_q_loss.mean() + (
                            mixer_weight / total_weight
                        ) * min_q_loss_mixer.mean()
                    else:
                        q_loss_both = min_q_loss.mean() + min_q_loss_mixer.mean()
                    qf_loss = qf_loss + q_loss_both

                elif self.inverse_weight:
                    # directly penalise without learning alphas as per the lagrange, but inverted
                    mixer_weight = (1 / (min_q_loss_mixer.mean() + 1e-12)).detach()
                    indep_weight = (1 / (min_q_loss.mean() + 1e-12)).detach()

                    print(mixer_weight.shape, indep_weight.shape)
                    total_weight = mixer_weight + indep_weight
                    qf_loss = (
                        qf_loss
                        + (mixer_weight / total_weight) * min_q_loss_mixer.mean()
                        + (indep_weight / total_weight) * min_q_loss.mean()
                    )
                else:
                    # directly penalise without learning alphas as per the lagrange
                    qf_loss = qf_loss + min_q_loss_mixer.mean() + min_q_loss.mean()

            else:
                raise Exception("Not a valid mixer, cql, cql_double combination")

            """
            Update networks
            """
            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            total_qf_loss.append(ptu.get_numpy(qf_loss))
            total_y_pred.append(ptu.get_numpy(y_pred))

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.soft_target_tau)
            if self.mixer is not None:
                ptu.soft_update_from_to(self.mixer, self.target_mixer, self.soft_target_tau)

        """
        Save some statistics for eval using just one batch.
        These are all wrong...maybe rework them in a bit.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics["QF Loss"] = np.mean(total_qf_loss)
            try:
                self.eval_statistics.update(
                    create_stats_ordered_dict(
                        "Y Predictions",
                        total_y_pred,
                    )
                )
            except:
                pass

        self._n_train_steps_total += 1
