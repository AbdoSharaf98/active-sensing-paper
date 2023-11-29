"""
A script for training the perception model separately
"""
from copy import deepcopy

from annealing_schedules import step_schedule, linear_cyclical_schedule
from envs.active_sensing import mnist_active_sensing, cifar_active_sensing

from models import perception
from models.perception import PerceptionModel
from models.action import RandomActionStrategy

from utils.data import *
from utils.training import train_perception_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# create the environment
config = deepcopy(mnist_active_sensing.DEFAULT_CONFIG)
config['batch_size'] = -1     # use entire dataset
config['n_samples'] = n = 5
config['sample_dim'] = d = 6
config['num_foveated_patches'] = nfov = 1
config['fovea_scale'] = fovsc = 2
config['num_workers'] = 0
config['valid_frac'] = 0.0
config['dataset'] = get_fashion_mnist()
env = mnist_active_sensing.make_env(config)

# get data from the environment
action_grid_sz = (33, 33)
data_dict = collect_data(env, actor=RandomActionStrategy(None, action_grid_sz, discrete=False))

# create or load the perception model
z_dim = 64
s_dim = 128
d_action = 2
d_obs = env.observation_space.shape[-1]

# create the perception model
vae_params = perception_v2.DEFAULT_PARAMS.copy()
vae_params['lower_vae']['layers'] = [256, 256]
vae_params['summarization_method']: 'cat'
vae_params['higher_vae'] = {
        'layers': [256, 256],
        'integration_method': 'sum',
        'rnn_hidden_size': 512,
        'rnn_num_layers': 1,
        'seq_len': n+1
    }


perception_model = PerceptionModel(z_dim, s_dim, d_action, d_obs, vae_params=vae_params,
                                   lr=0.001, encode_loc=False, use_latents=True).to(device)

# perception model training
log_dir = f'../perception_runs/fashion_mnist/n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_re'

# training params
epochs = 10
batch_size = 64
beta_sched = 1.1 * np.ones((epochs,)) #linear_cyclical_schedule(epochs, stop=0.25)
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

