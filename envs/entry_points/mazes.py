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
from utils.mazes import *
import string


def make_maze_prob_table(maze, N, M, gwells, uniforms=[], deterministic=False):
    prob_table = np.zeros((N, M, N))
    cols = (len(maze[0]) - 1) / 4
    curr = 0
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if type(maze[i][j]) != int:
                continue
            neighbors = []

            # for each direction
            for x, y in ((0, -1), (1, 0), (0, 1), (-1, 0)):

                # Teleporters are 1 space away
                if str(maze[i + y][j + x]) in 't':
                    neighbors.append(int(gwells[maze[i + y][j + x]]))

                # Walls are 2 spaces away
                elif maze[i + y * 2][j + x * 2] == 'w':
                    neighbors.append(int(curr))

                # Neighbors are 4 spaces away
                elif type(maze[i + y * 4][j + x * 4]) == int:
                    neighbors.append(int(curr + 1 * x + cols * y))
                else:
                    raise Exception("Malformed Maze, pos=%d" % curr)

            uniform = curr in uniforms
            curr_dist = assign_dist_to_state(M, uniform=uniform, deterministic=deterministic)
            for curr_a in range(len(curr_dist)):
                for n, curr_n in enumerate(neighbors):
                    prob_table[curr, curr_a, curr_n] += curr_dist[curr_a][n]

            curr = curr + 1

    return prob_table


class Maze(gym.Env, ABC):

    def __init__(self, w, h, num_actions, deterministic=False):
        # generate the maze
        self.maze_str = make_maze(w, h)

        # parse the maze
        self.maze, num_states, uniforms, gwells = parse_maze(self.maze_str)

        # define the observation and action spaces
        obs_shape = (num_states,)
        self.observation_space = Discrete(n=num_states)
        self.action_space = Discrete(n=num_actions)

        # create the probability table
        self.prob_table = make_maze_prob_table(self.maze, num_states, num_actions, gwells,
                                               uniforms=uniforms, deterministic=deterministic)

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
        super(Maze, self).seed(seed)
