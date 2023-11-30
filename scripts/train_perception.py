from envs.active_sensing import active_sensing_env

from models import perception
from models.perception import PerceptionModel
from models.action import RandomActionStrategy

from utils.data import *
from utils.training import train_perception_model

import argparse
import yaml


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./runs/perception_pretraining")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config_dir", type=str, default="./configs/bas.yaml")
    parser.add_argument("--env_config_dir", type=str, default="./configs/envs.yaml")
    parser.add_argument("--env_name", type=str, default="mnist")
    parser.add_argument("--discrete_action", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rec_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main(args):
    # create the environment
    with open(args.env_config_dir, "r") as f:
        env_config = yaml.safe_load(f)[args.env_name]
    env_config['batch_size'] = -1
    env_config['valid_frac'] = 0

    if args.env_name == "mnist":
        env_config['dataset'] = get_mnist_data()
    elif args.env_name == "translated_mnist":
        env_config['dataset'] = get_mnist_data(data_version="translated")
    elif args.env_name == "fashion_mnist":
        env_config['dataset'] = get_fashion_mnist()
    else:
        env_config['dataset'] = get_cifar()

    n, d = env_config['n_samples'], env_config['sample_dim']
    nfov, fovsc = env_config['num_foveated_patches'], env_config['fovea_scale']

    env = active_sensing_env.make_env(env_config)

    # set seed
    seed = args.seed if args.seed is not None else np.random.randint(999)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # collect data from the environment
    action_grid_sz = (33, 33)
    data_dict = collect_data(env, actor=RandomActionStrategy(None, action_grid_sz, discrete=args.discrete_action))

    # create perception model
    with open(args.config_dir, "r") as f:
        model_config = yaml.safe_load(f)['perception_model']
    model_config['obs_dim'] = env.observation_space.shape[-1]
    model_config['vae_params']['higher_vae']['seq_len'] = n + 1
    perception_model = PerceptionModel(**model_config).to(args.device)

    # logging directory
    # experiment name
    exp_name = f"n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_{seed}" if args.exp_name is None else args.exp_name

    # log dir
    log_dir = os.path.join(args.log_dir, args.env_name, exp_name)

    # training
    beta_sched = args.beta * np.ones((args.num_epochs,))
    train_perception_model(perception_model,
                           n_epochs=args.num_epochs,
                           batch_size=args.batch_size,
                           data=data_dict,
                           log_dir=log_dir,
                           exp_name='',
                           beta_schedule=beta_sched,
                           rec_loss_scale=args.rec_scale,
                           train_size=len(data_dict['obs']),
                           dataset_class=ActiveSensingDataset,
                           monitored_loss='total_loss')


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
