import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR, MultiStepLR
from agents.cmc_explorer import CMCExplorer
from envs.entry_points.worlds import World, DenseWorld, World123
from envs.entry_points.mazes import Maze
import numpy as np
import argparse
import os
import yaml


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./runs/cmc_explorer")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config_dir", type=str, default="./configs/cmc.yaml")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--env", type=str, default="maze")
    parser.add_argument("--total_steps", type=int, default=3000)
    parser.add_argument("--learning_starts", type=int, default=0)
    parser.add_argument("--learn_every", type=int, default=1)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main(args):
    # create a logging directory
    seed = args.seed if args.seed is not None else np.random.randint(999)
    exp_name = args.exp_name if args.exp_name is not None else seed
    logdir = os.path.join(args.log_dir, str(exp_name))
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # parse the config file
    with open(args.config_dir, "r") as f:
        config = yaml.safe_load(f)

    # create the environment
    if args.env == "maze":
        env = Maze(**config['env']['maze'])
    elif args.env == "dense_world":
        env = DenseWorld(**config['env']['dense_world'])
    else:
        env = World123(**config['env']['world123'])

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # learning rate scheduler
    lr_sched = lambda optim: ExponentialLR(optim, gamma=args.lr_gamma)

    # create the models
    bas_explorer = CMCExplorer(env, **config['model']['bas'], lr_scheduler=lr_sched,
                               action_strategy='bas').to(args.device)
    boltzmann_explorer = CMCExplorer(env, **config['model']['boltzmann'], lr_scheduler=lr_sched,
                                     action_strategy='boltzmann').to(args.device)
    random_explorer = CMCExplorer(env, **config['model']['random'], lr_scheduler=lr_sched,
                                  action_strategy='random').to(args.device)

    missing_infos_bas, visited_bas = bas_explorer.learn(total_steps=args.total_steps,
                                                        learning_starts=args.learning_starts,
                                                        learn_every=args.learn_every,
                                                        validate_every=args.validate_every,
                                                        prefix=f"BAS ({seed})")
    print('')

    missing_infos_boltz, visited_boltz = boltzmann_explorer.learn(total_steps=args.total_steps,
                                                                  learning_starts=args.learning_starts,
                                                                  learn_every=args.learn_every,
                                                                  validate_every=args.validate_every,
                                                                  prefix=f"Boltzmann ({seed})")

    print('')

    missing_infos_random, visited_random = random_explorer.learn(total_steps=args.total_steps,
                                                                 learning_starts=args.learning_starts,
                                                                 learn_every=args.learn_every,
                                                                 validate_every=args.validate_every,
                                                                 prefix=f"Random ({seed})")
    print('')

    # save
    to_save = {"maze_array": np.array(env.maze) if args.env == "maze" else None,
               "true_dist": env.prob_table,
               "random": {
                   "learned_dist": random_explorer.learned_dist,
                   "missing_info": missing_infos_random,
                   "visited": visited_random,
                   "history": random_explorer.history
               },
               "bas": {
                   "learned_dist": bas_explorer.learned_dist,
                   "missing_info": missing_infos_bas,
                   "visited": visited_bas,
                   "history": bas_explorer.history
               },
               "boltzmann": {
                   "learned_dist": boltzmann_explorer.learned_dist,
                   "missing_info": missing_infos_boltz,
                   "visited": visited_boltz,
                   "history": boltzmann_explorer.history
               }}
    torch.save(to_save, os.path.join(logdir, f"data_{seed}"))


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
