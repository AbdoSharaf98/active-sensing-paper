import warnings
from abc import ABC
import gym
from typing import Optional
import numpy as np
from gym.spaces import Box, Discrete
import scipy as scp
from scipy.special import comb
import math
import random


# Find the probability we get an edge to the absorber node
def prob_absorb(N, a):
    num = 1 - math.pow(0.75, a + 1)
    den = float(comb(N - 1, a))
    return num / den


class Node:
    # Given r > 0, r <= 1, find which target node we should make an edge to.
    # Return False if we find a target we already have
    def find_target(self, action, pabsorb, pnormal, r):
        for to in range(1, self.N):
            x = pabsorb + ((to) * pnormal)
            if x - r >= 0.0:
                if to in self.actions[action]:  # dupe
                    return False
                self.actions[action].append(to)
                return True
        x = 1 / 0

    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.actions = []
        absorber = 0

        for action in range(M):
            self.actions.append([])
            pabsorb = prob_absorb(N, action)
            pnormal = (1 - pabsorb) / (self.N - 1)
            nodes_left = action + 1
            while nodes_left > 0:
                r = round(random.random(), 2)
                if r < pabsorb:
                    if absorber in self.actions[action]:  # dupe
                        continue  # try again
                    self.actions[action].append(absorber)
                else:
                    if not self.find_target(action, pabsorb, pnormal, r):  # dupe
                        continue  # try again
                nodes_left -= 1

    def get_prob(self, a, ns):
        return 1.0 / (a + 1) if ns in self.actions[a] else 0

    def take_action(self, a):
        return random.sample(self.actions[a], 1)[0]


class World(gym.Env, ABC):

    def __init__(self, num_states: int, num_actions: int):
        self.prob_table = None  # to be defined in subclasses

        # define the observation and action spaces
        obs_shape = (num_states,)
        self.observation_space = Discrete(n=num_states)
        self.action_space = Discrete(n=num_actions)

        self.current_state = None

    def reset(self):
        # select a random state
        state_idx = np.random.randint(low=0, high=self.observation_space.n)
        self.current_state = state_idx

        return self.current_state

    def step(self, action: int):
        # get the probability distribution
        if self.prob_table is None:
            raise NotImplementedError
        transition_dist = self.prob_table[self.current_state][action]

        # sample the next state
        next_state = np.argmax(np.random.multinomial(1, transition_dist, size=1).squeeze())

        # set the current state
        self.current_state = next_state

        # return
        return self.current_state, 0.0, False, {}
    
    def seed(self, seed=None):
        super(World, self).seed(seed)


class DenseWorld(World, ABC):

    def __init__(self, num_states: int, num_actions: int, alphas: np.ndarray = None):
        super(DenseWorld, self).__init__(num_states, num_actions)

        if alphas is None:
            alphas = np.ones((num_states, ))

        self.prob_table = scp.stats.dirichlet.rvs(alphas, size=num_states * num_actions).reshape(num_states,
                                                                                                 num_actions,
                                                                                                 num_states)


class World123(World, ABC):

    def __init__(self, num_states: int, num_actions: int):
        super(World123, self).__init__(num_states, num_actions)

        nodes = [Node(num_actions, num_states) for _ in range(num_states)]

        # create the probability table
        self.prob_table = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                for sp in nodes[s].actions[a]:
                    self.prob_table[s, a, sp] = 1/len(nodes[s].actions[a])


