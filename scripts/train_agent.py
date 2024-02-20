import torch
import os
from warnings import warn
import numpy as np
import yaml

from envs.active_sensing import active_sensing_env
from utils.data import get_mnist_data, get_fashion_mnist, get_kmnist_data, get_cifar

from models.perception import PerceptionModel

from models.action import ActionNetworkStrategy, DirectEvaluationStrategy, RandomActionStrategy

from nets import ConcatDecisionNetwork, FFDecisionNetwork, RNNDecisionNetwork

from agents.active_sensor import BayesianActiveSensor

import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="../runs")
    parser.add_argument("--exp_name", type=str, default="bas_rnn_n=12_d=12_412")
    parser.add_argument("--seed", type=int, default=412)
    parser.add_argument("--config_dir", type=str, default="../configs/bas.yaml")
    parser.add_argument("--env_config_dir", type=str, default="../configs/envs.yaml")
    parser.add_argument("--env_name", type=str, default="cluttered_mnist")
    parser.add_argument("--perception_path", type=str,
                        default="../runs/perception_pretraining/cluttered_mnist/n=6_d=12_nfov=3_fovsc=2_412/last.ckpt")
    parser.add_argument("--load_model", type=str,
                        default=None)
    parser.add_argument("--action_strategy", type=str, default="bas")
    parser.add_argument("--decision_strategy", type=str, default="rnn")
    parser.add_argument("--finetune_e2e", default=False, action="store_true")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_warmup_epochs", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
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

    # create a perception model if no pre-trained model is specified
    if args.perception_path is None:
        warn("No pretrained perception model is provided. Creating a new perception model...")
        perception_config = model_config['perception_model']
        perception_config['obs_dim'] = env.observation_space.shape[-1]
        perception_config['vae_params']['higher_vae']['seq_len'] = n + 1
        perception_model = PerceptionModel(**perception_config).to(args.device)
    else:
        perception_model = PerceptionModel.load_from_checkpoint(args.perception_path).to(args.device)

    # create the actor
    action_config = model_config['action_model']
    if args.action_strategy == "bas":
        actor = ActionNetworkStrategy(perception_model, layers=action_config['layers'],
                                      lr=action_config['lr'],
                                      out_dist='gaussian',
                                      action_std=action_config['action_std'])
    elif args.action_strategy == "random":
        actor = RandomActionStrategy(perception_model)  # for random action strategy
    else:
        raise parser.error(message="Invalid action strategy. Valid input can be one of "
                                   "['bas', 'random']")

    # create the deciders
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
    elif args.decision_strategy == "perception":
        decider = FFDecisionNetwork(perception_model.s_dim,
                                    decision_config['layers'],
                                    env.num_classes,
                                    lr=decision_config['lr']).to(args.device)
    else:
        raise parser.error(message="Invalid decision strategy. Valid input can be one of "
                                   "['rnn', 'concat', 'perception']")

    # logging and checkpointing directory
    # experiment name
    exp_name = f"n={n}_d={d}_nfov={nfov}_fovsc={fovsc}_{seed}" if args.exp_name is None else args.exp_name
    # log dir
    log_dir = os.path.join(args.log_dir, args.env_name, exp_name)

    # build the active sensor model
    active_sensor = BayesianActiveSensor(env, perception_model, actor, decider,
                                         log_dir=log_dir, checkpoint_dir=log_dir,
                                         e2e_finetuning=args.finetune_e2e,
                                         device=args.device, decider_input=args.decision_strategy)

    if args.load_model is not None:
        active_sensor.load_from_checkpoint_dict(torch.load(args.load_model))

    # train
    beta_sched = args.beta * np.ones((args.num_epochs,))
    active_sensor.learn(num_epochs=args.num_epochs,
                        beta_sched=beta_sched,
                        num_random_epochs=args.num_warmup_epochs,
                        validate_every=args.validate_every)


if __name__ == "__main__":
    main(get_arg_parser())
