import torch
import os
from copy import deepcopy

from envs.active_sensing import mnist_active_sensing
from models.actors import RandomActionStrategy
from utils.data import get_mnist_data

from ram.model import RecurrentAttentionModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# create the environment
config = deepcopy(mnist_active_sensing.DEFAULT_CONFIG)
config['batch_size'] = 80
config['val_batch_size'] = 1000
config['n_samples'] = n = 3
config['sample_dim'] = d = 8
config['num_foveated_patches'] = 1
config['fovea_scale'] = 1
config['num_workers'] = 0
config['valid_frac'] = 0.1
config['dataset'] = get_mnist_data()
env = mnist_active_sensing.make_env(config)

# create the model
h_g = 32
h_l = 32
hidden_size = 64
std = 1.0
M = 1
log_dir = '../runs/ram_n=4_d=8_nfov=1_fovsc=1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_strategy = RandomActionStrategy(None, action_grid_size=(15, 15))
ram_model = RecurrentAttentionModel(env, h_g, h_l, std, hidden_size, random_strategy=random_strategy,
                                    log_dir=log_dir, discrete_loc=True)

# train
n_epochs = 200
ram_model.learn(num_epochs=n_epochs,
                M=M,
                num_random_epochs=1,
                validate_every=0.05)
