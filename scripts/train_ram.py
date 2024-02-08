import torch
import os
from copy import deepcopy

from envs.active_sensing import active_sensing_env
from models.action import RandomActionStrategy
from utils.data import get_mnist_data, get_fashion_mnist, get_cifar

from ram.model import RecurrentAttentionModel
import numpy as np
import yaml
import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./runs/ram")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config_dir", type=str, default="./configs/ram.yaml")
    parser.add_argument("--env_config_dir", type=str, default="./configs/envs.yaml")
    parser.add_argument("--env_name", type=str, default="mnist")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_warmup_epochs", type=int, default=0)
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--validate_every", type=float, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main(args):
    # create the environment
    with open(args.env_config_dir, "r") as f:
        env_config = yaml.safe_load(f)[args.env_name]

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

    # load the config file
    with open(args.config_dir, "r") as f:
        model_config = yaml.safe_load(f)['ram']

    # logging and checkpointing directory
    # experiment name
    exp_name = f"n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_{seed}" if args.exp_name is None else args.exp_name
    # log dir
    log_dir = os.path.join(args.log_dir, args.env_name, exp_name)

    random_strategy = RandomActionStrategy(None)

    # create the model
    ram_model = RecurrentAttentionModel(env, **model_config, random_strategy=random_strategy,
                                        log_dir=log_dir, discrete_loc=False).to(args.device)

    # train
    ram_model.learn(num_epochs=args.num_epochs,
                    M=args.M,
                    num_random_epochs=args.num_warmup_epochs,
                    validate_every=args.validate_every)


if __name__ == "__main__":
    parser = get_arg_parser()
    main(parser.parse_args())
