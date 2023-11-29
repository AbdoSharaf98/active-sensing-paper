from typing import Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
from envs.entry_points.worlds import World, DenseWorld, World123
from envs.entry_points.mazes import Maze
from nets import create_ff_network
import numpy as np
from torch.distributions import Dirichlet, Categorical
from torch.distributions.kl import kl_divergence
import os

torch.autograd.set_detect_anomaly(True)

EPS = 1e-20


def to_one_hot(idx: int, num_classes: int):
    one_hot = np.zeros((num_classes,))
    one_hot[idx] = 1
    return one_hot


class CMCExplorer(nn.Module):

    def __init__(self, env: Optional[World],
                 h_layers: list[int] = None,
                 action_strategy: str = 'bas',  # can be one of ['bas', 'random', 'boltzmann']
                 lr=0.001,
                 action_lr=0.05,
                 tau=0.2,
                 beta_action=1.0,
                 beta_perception=0.001,
                 lr_scheduler=None,
                 info_objective=None,
                 device='cuda'):

        super(CMCExplorer, self).__init__()

        self.env = env
        self.learned_dist = None
        self.info_objective = info_objective

        self.num_states, self.num_actions = env.observation_space.n, env.action_space.n
        self.hot_states = nn.functional.one_hot(torch.arange(self.num_states), self.num_states).to(device)
        self.hot_actions = nn.functional.one_hot(torch.arange(self.num_actions), self.num_actions).to(device)
        self.rep_states = torch.reshape(self.hot_states,
                                        (self.num_states,
                                         1, self.num_states)).repeat(1, self.num_actions, 1).flatten(end_dim=1)
        self.rep_actions = self.hot_actions.reshape((1, self.num_actions,
                                                     self.num_actions)).repeat(self.num_states, 1, 1).flatten(end_dim=1)

        # history tensor
        self.history = torch.zeros((self.num_states, self.num_actions, self.num_states),
                                   requires_grad=False, device=device)

        # build the encoder
        if h_layers is None:
            h_layers = [64, 64]
        enc_layers = [2 * self.num_states + self.num_actions] + h_layers + [self.num_states]
        self.encoder = create_ff_network(enc_layers, h_activation='none', out_activation='softplus')

        # build the action network
        self.action_network = create_ff_network([self.num_states] + h_layers + [self.num_actions], h_activation='relu',
                                                out_activation='softmax')
        self.tau = tau
        self.beta_action = beta_action
        self.beta_perception = beta_perception

        # store the action strategy
        self.action_strategy = action_strategy

        # build the optimizer
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.action_optimizer = torch.optim.Adam(self.action_network.parameters(), lr=action_lr)

        # learning rate scheduler
        if lr_scheduler is not None:
            self.scheduler = lr_scheduler(self.optimizer)
        else:
            self.scheduler = lr_scheduler

        # move to device
        self.device = device
        self.to(device)

    def reset_history(self):

        self.history = torch.zeros((self.num_states, self.num_actions, self.num_states),
                                   requires_grad=False, device=self.device)

    def encode(self, state, action, one_hot=False, hist=None):
        if one_hot:
            action_idx = torch.argmax(action).item()
            state_idx = torch.argmax(state).item()
        else:
            action_idx, state_idx = action, state
            action = torch.tensor(to_one_hot(action, self.num_actions)).float().to(self.device)
            state = torch.tensor(to_one_hot(state, self.num_states)).float().to(self.device)
        if hist is None:
            hist = self.history[state_idx, action_idx]

        encoder_input = torch.cat([state, action, hist], dim=-1)
        alpha_hat = self.encoder(encoder_input) + EPS  # for numerical stability
        q_alpha = Dirichlet(alpha_hat)

        return q_alpha, q_alpha.rsample()

    def compute_entropy_score(self, state, action, one_hot=False, hist=None):

        if one_hot:
            action_idx = torch.argmax(action).item()
            state_idx = torch.argmax(state).item()
        else:
            action_idx, state_idx = action, state
            action = torch.tensor(to_one_hot(action, self.num_actions)).float().to(self.device)
            state = torch.tensor(to_one_hot(state, self.num_states)).float().to(self.device)

        if hist is None:
            hist = self.history[state_idx, action_idx]

        # compute the current estimate of transition distribution
        q_pre, z = self.encode(state, action, True, hist)

        # compute the expected reduction in entropy
        updated_hists = self.history[state_idx, action_idx].reshape((1, -1)).repeat(self.num_states,
                                                                                    1) + self.hot_states
        updated_qs, _ = self.encode(state.reshape((1, -1)).repeat(self.num_states, 1),
                                    action.reshape((1, -1)).repeat(self.num_states, 1),
                                    True,
                                    hist=updated_hists)
        score = torch.sum(q_pre.entropy() - z * updated_qs.entropy())

        # compute expected surprise
        future_hist = torch.reshape(updated_hists,
                                    (self.num_states,
                                     1, self.num_states)).repeat(1, self.num_actions, 1).flatten(end_dim=1)

        expected_entropies = self.encode(self.rep_states, self.rep_actions,
                                         one_hot=True, hist=future_hist)[0].entropy()
        future_uncertainty = torch.reshape(expected_entropies,
                                           (self.num_states, self.num_actions)).mean(dim=-1)
        expected_uncertainty = torch.sum(z * future_uncertainty)

        if self.info_objective == 'score':
            return score
        elif self.info_objective == 'future':
            return expected_uncertainty
        else:
            return score + self.beta_action * expected_uncertainty

    def _compute_loss(self, state, action):
        # encode with the current history
        q_alpha, z = self.encode(state, action)

        # compute the kl loss
        prior = Dirichlet(torch.ones_like(q_alpha.concentration).to(self.device))
        kl_loss = kl_divergence(q_alpha, prior)

        # compute the NLL (we normalize by the size of the history)
        nll = -torch.sum(self.history[state, action] * torch.log(z))
        nll = nll / torch.max(torch.sum(self.history[state, action]), torch.tensor(1.0).to(self.device))

        return nll, kl_loss

    def compute_loss(self):
        total_nll = 0
        total_kl = 0
        visited_pairs = 0
        for state in range(self.num_states):
            for action in range(self.num_actions):
                if torch.sum(self.history[state, action]).item() != 0:
                    nll, kl = self._compute_loss(state, action)
                    total_nll = total_nll + nll
                    total_kl = total_kl + kl
                    visited_pairs = visited_pairs + 1

        return total_nll, total_kl, visited_pairs

    def select_action_with_net(self, state, train=True):  # TODO: experimental (currently not being used)
        state = torch.tensor(to_one_hot(state, self.num_states)).float().to(self.device)
        action_logits = self.action_network(state)
        action = torch.nn.functional.gumbel_softmax(action_logits, self.tau, hard=True)
        score = self.compute_entropy_score(state, action, one_hot=True)

        if train:
            self.action_optimizer.zero_grad()
            (-score).backward()
            self.action_optimizer.step()

        return torch.argmax(action).item()

    def select_action(self, state, strategy=None, net=False, tau=None):

        if strategy is None:
            strategy = self.action_strategy

        if strategy == 'bas':
            best_score = -torch.inf
            best_action = None

            if not net:

                for a in range(self.num_actions):

                    score = self.compute_entropy_score(state, a)

                    if score >= best_score:
                        best_action = a
                        best_score = score
            else:
                best_action = self.select_action_with_net(state, train=True)
        elif strategy == 'boltzmann':
            if tau is None:
                tau = 0.5
            boltz_probs = torch.softmax(-torch.sum(self.history[state], dim=-1) / tau, 0)
            best_action = Categorical(boltz_probs).sample()
        else:
            best_action = np.random.choice(self.num_actions)

        return best_action

    def learn(self, total_steps=50000,
              learning_starts=1000,
              learn_every=200,
              validate_every=500,
              verbose=True,
              prefix='',
              tau_sched=None):

        # reset the environment
        state = self.env.reset()

        missing_infos = []
        visited = []
        lowest_misinfo = torch.inf

        global_step = 0
        while global_step < total_steps:

            # create a temperature schedule if strategy is boltzmann
            if tau_sched is None:
                tau_sched = torch.linspace(1.0, 0.1, total_steps - learning_starts + 1)

            # select an action
            action = self.select_action(state,
                                        strategy='random' if global_step < learning_starts else self.action_strategy,
                                        net=False, tau=tau_sched[global_step])
            # step the environment and get the next state
            next_state = self.env.step(action)[0]
            next_hot_state = torch.tensor(to_one_hot(next_state, self.num_states)).float().to(self.device)
            # store in history
            self.history[state, action] = self.history[state, action] + next_hot_state

            # learn
            if global_step >= learning_starts:
                if global_step % learn_every == 0:
                    nll, kl = self._compute_loss(state, action)
                    loss = nll + self.beta_perception * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=10.0)
                    self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

            # validate
            if global_step % validate_every == 0:
                if global_step >= learning_starts:
                    loss, kl, visited_pairs = self.compute_loss()
                    loss = loss.detach()
                    kl = kl.detach()
                else:
                    loss = torch.nan
                    kl = torch.nan
                    visited_pairs = torch.sum(torch.any(self.history)).item()
                learned_dist, _, missing_info = self.validate()
                if missing_info < lowest_misinfo:
                    self.learned_dist = learned_dist
                    lowest_misinfo = missing_info
                missing_infos.append(missing_info.item())
                visited.append(100 * visited_pairs / (self.num_states * self.num_actions))
                if verbose:
                    print(f"{prefix} \t Step {global_step}: Loss = {loss},"
                          f"\t KL = {kl}"
                          f" \tMissing Info = {missing_info}"
                          f" \tVisited = {visited_pairs}/{(self.num_states * self.num_actions)}", end='\r')

            state = next_state
            global_step = global_step + 1

        return missing_infos, visited

    def validate(self):

        learned_dist = torch.zeros((self.num_states, self.num_actions, self.num_states))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                learned_dist[s][a] = self.encode(s, a)[0].mean

        learned_dist = learned_dist.cpu().detach().numpy()
        true_dist = self.env.prob_table + EPS

        missing_info = np.sum(np.sum(true_dist * np.log(true_dist / learned_dist), axis=-1))

        return learned_dist, self.env.prob_table, missing_info
