from typing import Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
from envs.entry_points.worlds import World, DenseWorld, World123
from envs.entry_points.mazes import Maze
from nets import create_ff_network
import numpy as np
from torch.nn import functional as F
from torch.distributions import Dirichlet, Categorical
from torch.distributions.kl import kl_divergence
import os

import matplotlib.pyplot as plt
import scienceplots

torch.autograd.set_detect_anomaly(True)

EPS = 1e-20


def to_one_hot(idx: int, num_classes: int):
    one_hot = np.zeros((num_classes,))
    one_hot[idx] = 1
    return one_hot


class CMCExplorer(nn.Module):

    def __init__(self, env: Optional[World],
                 h_layers: list[int] = None,
                 action_strategy: str = 'bas',  # action strategy can be either 'bas' or 'random'
                 lr=0.001,
                 action_lr=0.05,
                 tau=0.2,
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

        #future_hist = self.history.clone()
        #future_hist = future_hist.flatten(end_dim=1)

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
            return score + 0.01 * expected_uncertainty

    def two_step_entropy_score(self, state, action):

        # compute the current estimate of transition distribution
        q_pre, z = self.encode(state, action)

        # compute the expected reduction in entropy
        one_step_score = torch.tensor(0.0).float()
        two_step_score = torch.tensor(0.0).float()
        for s in range(self.num_states):
            next_state = torch.tensor(to_one_hot(s, self.num_states)).float().to(self.device)
            updated_hist = self.history[state, action] + next_state
            updated_q, _ = self.encode(state, action, hist=updated_hist)
            one_step_score = one_step_score + z[s] * (q_pre.entropy() - updated_q.entropy())
            for a in range(self.num_actions):
                two_step_score = two_step_score + self.compute_entropy_score(s, a, updated_hist)

        return one_step_score + two_step_score

    def compute_loss_(self, state, action):
        # encode with the current history
        q_alpha, z = self.encode(state, action)

        # compute the kl loss
        # prior = Dirichlet(torch.ones_like(q_alpha.concentration).to(self.device))
        kl_loss = torch.zeros(1)  # kl_divergence(q_alpha, prior)

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
                    nll, kl = self.compute_loss_(state, action)
                    total_nll = total_nll + nll
                    total_kl = total_kl + kl
                    visited_pairs = visited_pairs + 1

        return total_nll, total_kl, visited_pairs

    def select_action_with_net(self, state, train=True):
        state = torch.tensor(to_one_hot(state, self.num_states)).float().to(self.device)
        action_logits = self.action_network(state)
        action = torch.nn.functional.gumbel_softmax(action_logits, self.tau, hard=True)
        score = self.compute_entropy_score(state, action, one_hot=True)

        if train:
            self.action_optimizer.zero_grad()
            (-score).backward()
            self.action_optimizer.step()

        return torch.argmax(action).item()

    def select_action(self, state, two_step=False, strategy=None, net=False, tau=None):

        if strategy is None:
            strategy = self.action_strategy

        if strategy == 'bas':
            best_score = -torch.inf
            best_action = None

            if not net:

                for a in range(self.num_actions):

                    if two_step:
                        score = self.two_step_entropy_score(state, a)
                    else:
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
              beta=0.00, tau_sched=None):

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
            action = self.select_action(state, two_step=False,
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
                    nll, kl = self.compute_loss_(state, action)
                    loss = nll  # + beta * kl

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
                if self.action_strategy == 'bas':
                    learned_dist[s][a] = self.encode(s, a)[0].mean
                else:
                    learned_dist[s][a] = self.encode(s, a)[0].mean

        learned_dist = learned_dist.cpu().detach().numpy()
        true_dist = self.env.prob_table + EPS

        missing_info = np.sum(np.sum(true_dist * np.log(true_dist / learned_dist), axis=-1))

        return learned_dist, self.env.prob_table, missing_info


def plot_missing_info(bas_data, random_data, boltz_data, delta=10, labels=None):
    plt.style.use(['science', 'ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams.update({'font.size': 5})

    if labels is None:
        labels = ['BAS', 'Random', 'Boltzmann']

    bas_avg, bas_sem = np.mean(bas_data, 0), np.std(bas_data, 0) / np.sqrt(bas_data.shape[0])
    rnd_avg, rnd_sem = np.mean(random_data, 0), np.std(random_data, 0) / np.sqrt(random_data.shape[0])
    boltz_avg, boltz_sem = np.mean(boltz_data, 0), np.std(boltz_data, 0) / np.sqrt(boltz_data.shape[0])

    plt.plot(np.arange(len(bas_avg)) * delta, bas_avg, 'firebrick', label=labels[0])
    plt.plot(np.arange(len(rnd_avg)) * delta, rnd_avg, '-', color='navy', label=labels[1])
    plt.plot(np.arange(len(boltz_avg)) * delta, boltz_avg, '-', color='forestgreen', label=labels[2])
    plt.fill_between(np.arange(len(rnd_avg)) * delta, rnd_avg - rnd_sem, rnd_avg + rnd_sem, alpha=0.4, color='navy')
    plt.fill_between(np.arange(len(bas_avg)) * delta, bas_avg - bas_sem, bas_avg + bas_sem, alpha=0.4,
                     color='firebrick')
    plt.fill_between(np.arange(len(boltz_avg)) * delta, boltz_avg - boltz_sem, boltz_avg + boltz_sem, alpha=0.4,
                     color='forestgreen')
    plt.xlim([0, len(bas_avg) * delta])
    plt.xlabel('Step #')
    plt.ylabel('Missing Information (bits)')
    plt.ylim([0, max(max(np.max(bas_avg + bas_sem), np.max(rnd_avg + rnd_sem)), np.max(boltz_avg + boltz_sem)) + 20])
    plt.legend()


def prob_diff_heatmap(true_dist, learned_dist_bas, learned_dist_rand, science_style=False):
    if science_style:
        plt.style.use(['science', 'ieee'])
        plt.rcParams['text.usetex'] = False
        plt.rcParams.update({'font.size': 4})

    diff_dist_bas = np.abs(true_dist - learned_dist_bas)
    diff_dist_rand = np.abs(true_dist - learned_dist_rand)

    fig, axs = plt.subplots(2, 4, sharey=True, sharex=True)
    for a in range(true_dist.shape[1]):
        dist_im_bas = diff_dist_bas[:, a, :]
        dist_im_rand = diff_dist_rand[:, a, :]
        dist_im_bas = (dist_im_bas - np.min(diff_dist_bas[:])) / (np.max(diff_dist_bas[:]) - np.min(diff_dist_bas[:]))
        dist_im_rand = (dist_im_rand - np.min(diff_dist_rand[:])) / (
                np.max(diff_dist_rand[:]) - np.min(diff_dist_rand[:]))

        axs[0][a].imshow(dist_im_bas, cmap='Blues', interpolation='none', vmin=0, vmax=1)
        axs[0][a].set_title(f"{np.sum(dist_im_bas):0.2f}")  # axs[0][a].set_title(f"a = {a+1}")
        axs[1][a].set_title(f"{np.sum(dist_im_rand):0.2f}")
        im = axs[1][a].imshow(dist_im_rand, cmap='Blues', interpolation='none', vmin=0, vmax=1)
        axs[1][a].set_xlabel('s')

    axs[0][0].set_ylabel("s'")
    axs[1][0].set_ylabel("s'")

    plt.gcf().subplots_adjust(right=0.8)
    plt.gcf().colorbar(im, cax=plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7]))


def maze_heat_map(maze_array, visit_frequency, science_style=False):
    if science_style:
        plt.style.use(['science', 'ieee'])
        plt.rcParams['text.usetex'] = False

    # convert the maze array to a float array
    maze_array[maze_array == 'w'] = 0.1
    maze_array[maze_array == '.'] = 0.001
    maze_array = np.array(maze_array, dtype=np.float)

    # populate the maze array
    for st in range(len(visit_frequency)):
        maze_array[maze_array == st] = visit_frequency[st] / np.max(visit_frequency)

    plt.figure()
    plt.imshow(maze_array, cmap='hot', interpolation='kaiser')
    plt.colorbar(label='Visitation Frequency')

    return maze_array


if __name__ == "__main__":

    bas_info, random_info, boltzmann_info = [], [], []
    bas_visit, random_visit, boltzmann_visit = [], [], []
    logdir = f'../runs/cmc_explorer_maze_6x6_boltz'
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    N = 1
    seeds = np.random.choice(np.arange(100, 9999), replace=False, size=(N,))

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        dense_env = DenseWorld(num_states=10, num_actions=4, alphas=np.ones((10,)))
        env_123 = World123(num_states=20, num_actions=3)
        maze = Maze(6, 6, 4, deterministic=False)

        dense_env.seed(seed)
        env_123.seed(seed)
        maze.seed(seed)

        lr_sched = lambda optim: ExponentialLR(optim, gamma=1.0)

        bas_explorer = CMCExplorer(maze, h_layers=[16, 16], lr=0.001, lr_scheduler=lr_sched,
                                   action_strategy='bas', info_objective='score').cuda()
        boltzmann_explorer = CMCExplorer(maze, h_layers=[16, 16], lr=0.001, lr_scheduler=lr_sched,
                                         action_strategy='boltzmann').cuda()
        random_explorer = CMCExplorer(maze, h_layers=[16, 16], lr=0.001, lr_scheduler=lr_sched,
                                      action_strategy='random').cuda()

        missing_infos_bas, visited_bas = bas_explorer.learn(total_steps=3000, learning_starts=0, learn_every=1,
                                                            validate_every=10,
                                                            beta=0.005,
                                                            prefix=f"BAS ({seed})")
        print('')

        missing_infos_boltz, visited_boltz = boltzmann_explorer.learn(total_steps=3000, learning_starts=0,
                                                                      learn_every=1,
                                                                      validate_every=10,
                                                                      beta=0.005,
                                                                      prefix=f"BOLTZ ({seed})")

        print('')

        missing_infos_random, visited_random = random_explorer.learn(total_steps=3000, learning_starts=0,
                                                                     learn_every=1,
                                                                     validate_every=10,
                                                                     beta=0.005,
                                                                     prefix=f"RANDOM ({seed})")
        print('')

        # save
        to_save = {"maze_array": np.array(maze.maze),
                   "true_dist": maze.prob_table,
                   "random": {
                       "learned_dist": random_explorer.learned_dist,
                       "missing_info": missing_infos_random,
                       "visited": visited_random,
                       "history": random_explorer.history
                   },
                   "bas": {
                       "learned_dist": bas_explorer.learned_dist,
                       "missing_info": missing_infos_bas,
                       "visited": visited_bas,
                       "history": bas_explorer.history
                   },
                   "boltzmann": {
                       "learned_dist": boltzmann_explorer.learned_dist,
                       "missing_info": missing_infos_boltz,
                       "visited": visited_boltz,
                       "history": boltzmann_explorer.history
                   }}
        torch.save(to_save, os.path.join(logdir, f"data_{seed}"))

        bas_info.append(missing_infos_bas)
        random_info.append(missing_infos_random)
        boltzmann_info.append(missing_infos_boltz)
        bas_visit.append(visited_bas)
        random_visit.append(visited_random)
        boltzmann_visit.append(visited_boltz)

    # plot_missing_info(np.array(bas_info), np.array(boltzmann_info))

    print("Done")
