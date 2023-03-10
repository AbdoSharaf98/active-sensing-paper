import torch
import os
from copy import deepcopy
import numpy as np

from envs.active_sensing import mnist_active_sensing
from utils.data import get_mnist_data

from models import perception
from models.perception import PerceptionModel

from models.actors import ActionNetworkStrategy, DirectEvaluationStrategy, RandomActionStrategy

from nets import DecisionNetwork, FFDecisionNetwork, RNNDecisionNetwork

from agents.active_sensor import BayesianActiveSensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# create the environment
config = deepcopy(mnist_active_sensing.DEFAULT_CONFIG)
config['batch_size'] = 64
config['val_batch_size'] = 60
config['n_samples'] = n = 3
config['sample_dim'] = d = 8
config['num_foveated_patches'] = nfov = 1
config['fovea_scale'] = fovsc = 1
config['num_workers'] = 0
config['valid_frac'] = 0.1
config['dataset'] = get_mnist_data()
env = mnist_active_sensing.make_env(config)

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

# create the actor
action_grid_size = (9, 9)
# actor = ActionNetworkStrategy(perception_model, action_grid_size, layers=[32, 16], lr=0.001, out_dist='concrete')
actor = DirectEvaluationStrategy(perception_model, action_grid_size)


# create the decider
decision_mode = 'perception'  # options: 'perception', 'raw'
h_layers = [256, 256]
decider = FFDecisionNetwork(perception_model.s_dim,
                            h_layers,
                            env.num_classes,
                            lr=0.001).to(device)

# create the active sensor model
log_dir = f'../runs/bas_perception_n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_DirectEvaluation'
active_sensor = BayesianActiveSensor(env, perception_model, actor, decider,
                                     log_dir=log_dir, checkpoint_dir=log_dir,
                                     device=device, decider_input=decision_mode)

# train
n_epochs = 200
active_sensor.learn(num_epochs=n_epochs, beta_sched=np.ones((n_epochs,))*0.1, num_random_epochs=1,
                    validate_every=0.05)
