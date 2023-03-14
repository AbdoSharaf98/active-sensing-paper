"""
A script for training the perception model separately
"""

import torch
import numpy as np
import os
from copy import deepcopy

from annealing_schedules import step_schedule
from envs.active_sensing import mnist_active_sensing

from models import perception
from models.perception import PerceptionModel
from models.action import RandomActionStrategy

from utils.data import collect_data, ActiveSensingDataset, get_mnist_data, get_fashion_mnist
from utils.training import train_perception_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# create the environment
config = deepcopy(mnist_active_sensing.DEFAULT_CONFIG)
config['batch_size'] = -1     # use entire dataset
config['n_samples'] = n = 5
config['sample_dim'] = d = 8
config['num_foveated_patches'] = nfov = 1
config['fovea_scale'] = fovsc = 1
config['num_workers'] = 0
config['valid_frac'] = 0.0
config['dataset'] = get_fashion_mnist()
env = mnist_active_sensing.make_env(config)

# get data from the environment
action_grid_sz = (9, 9)
data_dict = collect_data(env, actor=RandomActionStrategy(None, action_grid_sz))

# create or load the perception model
z_dim = 32
s_dim = 64
d_action = 2
d_obs = env.observation_space.shape[-1]

# create the perception model
vae1_params = perception.DEFAULT_VAE1_PARAMS.copy()
vae1_params['type'] = 'mlp'
vae1_params['layers'] = [256, 256]

vae2_params = {
    'layers': [256, 256],
    'rnn_hidden_size': 512,
    'rnn_num_layers': 1
}

perception_model = PerceptionModel(z_dim, s_dim, d_action, d_obs, vae1_params=vae1_params,
                                   vae2_params=vae2_params, lr=0.001, use_latents=True).to(device)

# perception model training
log_dir = f'../perception_runs/perception_fashionMNIST_n={n}_d={d}_nfov={nfov}_fovsc={fovsc}'

# training params
epochs = 7
batch_size = 64
beta_sched = 0.1 * np.ones((epochs,))
rec_scale = 1.0

# train
train_perception_model(perception_model,
                       n_epochs=epochs,
                       batch_size=batch_size,
                       data=data_dict,
                       log_dir=log_dir,
                       exp_name='',
                       beta_schedule=beta_sched,
                       rec_loss_scale=rec_scale,
                       train_size=len(data_dict['obs']) - 10000,
                       dataset_class=ActiveSensingDataset,
                       monitored_loss='total_loss')

