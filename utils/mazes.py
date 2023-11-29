from random import shuffle, randrange
import random
import string
import math
import numpy as np


# Transform
def t(i):
    return 2 + 3 * i


# Place the highest probability in position a
def realign(M, a, dist):
    highest = -1.0
    high_i = -1
    for i in range(M):
        if dist[i] > highest:
            highest = dist[i]
            high_i = i

    # Swap dist[a] with dist[high_i]
    tmp = dist[a]
    dist[a] = highest
    dist[high_i] = tmp

    return dist


def assign_dist_to_state(M, uniform=False, deterministic=False):
    actions = []  # The distribution of each action
    for a in range(M):
        if deterministic:
            dist = [0] * a + [1] + [0] * (M - a - 1)  # Generate prob=1 for a
        elif not uniform:
            dist = np.random.dirichlet([1.0 / M] * M)  # Uniform alpha's
            dist = realign(M, a, dist)
        else:
            dist = [1.0 / M] * M  # Uniform

        actions.append(dist)

    return actions


def make_maze(w=10, h=10):
    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    space = [["w . . . "] * w + ['w'] for _ in range(h)] + [[]]
    border = [["w w w w "] * w + ['w'] for _ in range(h + 1)]
    locs = []

    num_tels = int(5 / 6.0 * w * h)

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]:
                continue
            if xx != x:
                space[y][max(x, xx)] = ". . . . "
            if yy != y:
                border[max(y, yy)][x] = "w . . . "
            walk(xx, yy)

    def number():
        g = random.randint(0, w * h)
        i = 0
        for s in range(len(space) - 1):
            locs.append([])
            for l in range(len(space[s]) - 1):
                arr = list(space[s][l])
                arr[4] = str(i % 10) if i != g else 'g'
                i = i + 1
                locs[s].append(''.join(arr))
            locs[s].append('w')
        locs.append([])

    walk(randrange(w), randrange(h))
    number()

    maze_string = ""
    for (a, b, c, d) in zip(border, space, locs, space):
        maze_string += (''.join(a + ['\n'] + b + ['\n'] + c + ['\n'] + d)) + '\n'

    return maze_string


def init_maze(l):
    w = math.ceil(len(l) / 2)
    maze = []
    for i in range(w):
        maze.append([])
        for j in range(w):
            maze[i].append(' ')
    return maze


def parse_maze(maze_str):
    maze_lines = maze_str.splitlines()

    l = maze_lines[0]
    maze = init_maze(l)

    nstates = 0
    uniforms = set()
    gwells = {}
    r = 0
    while l != '' and l != '\n':
        c = 0
        for i in l:
            if i == ' ' or i == '\n':
                continue
            if i == 'g':
                gwells[i.lower()] = nstates
                i = str(nstates % 10)

            if i == '!':
                uniforms.add(nstates)
                i = str(nstates % 10)

            if i in string.digits:
                nstates = nstates + 1
                i = nstates - 1

            maze[r][c] = i
            c = c + 1

        r = r + 1
        l = maze_lines[r]

    return maze, nstates, uniforms, gwells
