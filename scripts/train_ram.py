"""
A script for training the Recurrent Attention Model proposed in Mnih et al. https://arxiv.org/abs/1406.6247
"""


import torch
import os
from copy import deepcopy

from envs.active_sensing import mnist_active_sensing
from models.action import RandomActionStrategy
from utils.data import get_mnist_data, get_fashion_mnist

from ram.model import RecurrentAttentionModel
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

seeds = [66233]

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create the environment
    config = deepcopy(mnist_active_sensing.DEFAULT_CONFIG)
    config['batch_size'] = 64
    config['val_batch_size'] = 1000
    config['n_samples'] = n = 3
    config['sample_dim'] = d = 12
    config['num_foveated_patches'] = nfov = 3
    config['fovea_scale'] = fovsc = 2
    config['num_workers'] = 0
    config['valid_frac'] = 0.1
    config['dataset'] = get_mnist_data(data_version='translated')
    env = mnist_active_sensing.make_env(config)

    # create the model
    h_g = 64
    h_l = 64
    hidden_size = 128
    std = 0.05
    M = 1
    log_dir = f'../runs/translated_mnist/ram_baseline/n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_{seed}_re'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_strategy = RandomActionStrategy(None, action_grid_size=(9, 9))
    ram_model = RecurrentAttentionModel(env, h_g, h_l, std, hidden_size, random_strategy=random_strategy,
                                        log_dir=log_dir, discrete_loc=False)

    # train
    n_epochs = 50
    ram_model.learn(num_epochs=n_epochs,
                    M=M,
                    num_random_epochs=0,
                    validate_every=2)
