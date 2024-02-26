import torch
import os
from warnings import warn
import numpy as np
import yaml

from envs.active_sensing import active_sensing_env
from utils.data import get_mnist_data, get_fashion_mnist, get_kmnist_data, get_cifar

from agents import ppo_byol_explore as byolex

from models.action import ActionNetworkStrategy, DirectEvaluationStrategy, RandomActionStrategy

from nets import ConcatDecisionNetwork, FFDecisionNetwork, RNNDecisionNetwork

import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./runs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=412)
    parser.add_argument("--config_dir", type=str, default="./configs/byol.yaml")
    parser.add_argument("--env_config_dir", type=str, default="./configs/envs.yaml")
    parser.add_argument("--env_name", type=str, default="translated_mnist")
    parser.add_argument("--load_model", type=str,
                        default=None)
    parser.add_argument("--decision_strategy", type=str, default="concat")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_warmup_epochs", type=int, default=20)
    parser.add_argument("--validate_every", type=float, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main(parser):
    args = parser.parse_args()
    # create the environment
    with open(args.env_config_dir, "r") as f:
        env_config = yaml.safe_load(f)[args.env_name]

    if args.env_name == "mnist":
        env_config['dataset'] = get_mnist_data()
    elif args.env_name == "translated_mnist":
        env_config['dataset'] = get_mnist_data(data_version="translated")
    elif args.env_name == "fashion_mnist":
        env_config['dataset'] = get_fashion_mnist()
    elif args.env_name == "cluttered_mnist":
        env_config['dataset'] = get_mnist_data(data_version="cluttered")
    elif args.env_name == "cifar":
        env_config['dataset'] = get_cifar()
    else:
        raise

    n, d = env_config['n_samples'], env_config['sample_dim']
    nfov, fovsc = env_config['num_foveated_patches'], env_config['fovea_scale']

    env = active_sensing_env.make_env(env_config)

    # set seed
    seed = args.seed if args.seed is not None else np.random.randint(999)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load the config file
    with open(args.config_dir, "r") as f:
        model_config = yaml.safe_load(f)

    # create components of the BYOL-Explore agent
    # # encoders
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    latent_dim = model_config['encoders']['latent_dim']
    online_encoder, target_encoder, actor_encoder, critic_encoder = [byolex.ObservationEncoder(obs_dim, **model_config['encoders']) for _ in range(4)]
    
    # # actor critic
    actor_critic = byolex.ActorCritic(actor_encoder, critic_encoder, action_dim, **model_config['actor_critic'])
    
    # # world model
    world_model = byolex.WorldModel(latent_dim, action_dim, **model_config['world_model'])

    # # decider
    decision_config = model_config['decision_model']
    if args.decision_strategy == "rnn":
        decider = RNNDecisionNetwork(env.observation_space.shape[-1] + env.action_space.shape[-1],
                                     decision_config['layers'],
                                     env.num_classes,
                                     hidden_size=decision_config['rnn_hidden_size'],
                                     lr=decision_config['lr']).to(args.device)
    elif args.decision_strategy == "concat":
        decider = ConcatDecisionNetwork(env.observation_space.shape[-1] + env.action_space.shape[-1],
                                        seq_len=n + 1,
                                        layers=decision_config['layers'],
                                        num_classes=env.num_classes).to(args.device)
    else:
        raise parser.error(message="Invalid decision strategy. Valid input can be one of "
                                   "['rnn', 'concat']")

    # logging and checkpointing directory
    # experiment name
    exp_name = f"BYOLEX_{args.decision_strategy}_n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_{seed}" if args.exp_name is None else args.exp_name
    # log dir
    log_dir = os.path.join(args.log_dir, args.env_name, exp_name)

    # build the active sensor model
    byol_ppo = byolex.BYOLActiveSensor(env, online_encoder, target_encoder, actor_critic, world_model, decider,
                                       config=model_config['agent'], log_dir=log_dir, device=args.device).to(args.device)

    if args.load_model is not None:
        active_sensor.load_from_checkpoint_dict(torch.load(args.load_model))

    # train
    byol_ppo.learn(start_epoch=args.start_epoch,
                   num_epochs=args.num_epochs,
                   num_random_epochs=args.num_warmup_epochs,
                   validate_every=args.validate_every)


if __name__ == "__main__":
    main(get_arg_parser())
