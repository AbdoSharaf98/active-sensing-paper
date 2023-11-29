"""
A script for training the Bayesian Active Sensor
"""

import torch
import os
from copy import deepcopy
import numpy as np

from envs.active_sensing import mnist_active_sensing, cifar_active_sensing
from utils.data import get_mnist_data, get_fashion_mnist, get_kmnist_data, get_cifar

from models import perception
from models.perception import PerceptionModel

from models.action import ActionNetworkStrategy, DirectEvaluationStrategy, RandomActionStrategy

from nets import ConcatDecisionNetwork, FFDecisionNetwork, RNNDecisionNetwork

from agents.active_sensor import BayesianActiveSensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# random seed
seeds = [66232]
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    for action in ['bas']:
        for decision_mode in ['perception']:

            # create the environment
            config = deepcopy(mnist_active_sensing.DEFAULT_CONFIG)
            config['batch_size'] = 64
            config['val_batch_size'] = 1000
            config['n_samples'] = n = 5
            config['sample_dim'] = d = 6
            config['num_foveated_patches'] = nfov = 1
            config['fovea_scale'] = fovsc = 2
            config['num_workers'] = 0
            config['valid_frac'] = 0.1
            config['dataset'] = get_fashion_mnist()
            env = mnist_active_sensing.make_env(config)

            # create or load the perception model
            model_dir = f"../perception_runs/fashion_mnist/n=5_d=6_nfov=1_fovsc=2_re/last.ckpt"

            if model_dir is None:
                z_dim = 64
                s_dim = 128
                d_action = 2
                d_obs = env.observation_space.shape[-1]

                # create the perception model
                vae_params = perception.DEFAULT_PARAMS.copy()
                vae_params['lower_vae']['layers'] = [256, 256]
                vae_params['summarization_method']: 'cat'
                vae_params['higher_vae'] = {
                    'layers': [512, 512],
                    'integration_method': 'sum',
                    'rnn_hidden_size': 512,
                    'rnn_num_layers': 1
                }

                perception_model = PerceptionModel(z_dim, s_dim, d_action, d_obs, vae_params=vae_params,
                                                   lr=0.001, encode_loc=False, use_latents=True).to(device)
            else:
                perception_model = PerceptionModel.load_from_checkpoint(model_dir).to(device)

            # create the actor
            action_grid_size = (9, 9)

            bas_actor = ActionNetworkStrategy(perception_model, action_grid_size, layers=[64, 32], lr=0.001,
                                              out_dist='gaussian',
                                              action_std=0.05)
            random_actor = RandomActionStrategy(perception_model, action_grid_size, discrete=False)

            # create the decider
            h_layers = [256, 256]
            rnn_decider = RNNDecisionNetwork(env.observation_space.shape[-1] + env.action_space.shape[-1],
                                             h_layers,
                                             env.num_classes,
                                             hidden_size=64,
                                             lr=0.001).to(device)
            concat_decider = ConcatDecisionNetwork(env.observation_space.shape[-1] + env.action_space.shape[-1],
                                                   seq_len=n+1, layers=h_layers,
                                                   num_classes=env.num_classes).to(device)
            ff_decider = FFDecisionNetwork(perception_model.s_dim,
                                           h_layers,
                                           env.num_classes,
                                           lr=0.001).to(device)

            # create the active sensor model
            log_dir = f'../runs/fashion_mnist/n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_re/' \
                      f'{action}_{decision_mode}_{seed}'
            if action == 'bas':
                actor = bas_actor
            else:
                actor = random_actor
            if decision_mode == 'perception':
                decider = ff_decider
            else:
                decider = rnn_decider

            active_sensor = BayesianActiveSensor(env, perception_model, actor,
                                                 decider,
                                                 log_dir=log_dir, checkpoint_dir=log_dir,
                                                 device=device, decider_input=decision_mode)

            # train
            print(f"-------------------- {action} & {decision_mode} : {seed} -------------------")
            n_epochs = 80
            beta_sched = 1.1 * np.ones((n_epochs,))
            active_sensor.learn(num_epochs=n_epochs, beta_sched=beta_sched, num_random_epochs=0,
                                validate_every=4)
