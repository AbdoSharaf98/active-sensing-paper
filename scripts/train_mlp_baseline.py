import os.path

from colorama import Fore

from utils.data import get_mnist_data, create_data_loaders, get_cifar, get_fashion_mnist
from nets import create_ff_network
import torch
import numpy as np

import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./runs/mlp_baseline")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="translated_mnist")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--layers", nargs="*", type=int, default=[128, 128])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--validate_every", type=float, default=2.0)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main(args):

    # get the dataset and create the data loaders
    if args.dataset == "mnist":
        dataset = get_mnist_data()
    elif args.dataset == "translated_mnist":
        dataset = get_mnist_data(data_version="translated")
    elif args.dataset == "fashion_mnist":
        dataset = get_fashion_mnist()
    else:
        dataset = get_cifar()

    # set seed
    seed = args.seed if args.seed is not None else np.random.randint(999)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, valid_loader, test_loader = create_data_loaders(dataset, batch_size=args.batch_size,
                                                                  num_workers=0)

    # create the model
    in_dim, out_dim = np.prod(next(iter(train_loader))[0].shape[-2:]), len(dataset[0].classes)
    layers = [in_dim] + args.layers + [out_dim]
    model = create_ff_network(layers, h_activation='relu', out_activation='softmax').to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # logging directory
    # experiment name
    exp_name = f"{seed}" if args.exp_name is None else args.exp_name
    # log dir
    log_dir = os.path.join(args.log_dir, args.dataset, exp_name)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # train
    num_train = len(train_loader)
    validate_interval = int(args.validate_every * num_train)

    train_accs, valid_accs, test_accs = [], [], []

    max_v = 0.0
    max_t = 0.0
    total_updates = 0
    for epoch in range(args.num_epochs):
        epoch_accs = np.zeros((num_train,))
        for batch_num, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)

            # forward the model
            predicted = model(x.flatten(start_dim=1))

            # compute the loss
            loss = torch.nn.functional.cross_entropy(predicted, y)

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_updates += 1

            # compute the accuracy
            accuracy = np.sum((torch.argmax(predicted, dim=-1) == y).cpu().numpy()) / len(y)
            epoch_accs[batch_num] = accuracy

            # print
            print(Fore.YELLOW + f'[train] Episode {epoch + 1} ({batch_num + 1}/{num_train}):\t \033[1mSCORE\033['
                                f'0m = {accuracy:0.3f}'
                  + Fore.YELLOW + f' \t \033[1mAVG SCORE\033[0m = {epoch_accs[:batch_num + 1].mean():0.3f}',
                  end='\r')

            if total_updates % validate_interval == 0:
                with torch.no_grad():
                    # validation
                    vx, vy = next(iter(valid_loader))
                    vx, vy = vx.to(args.device), vy.to(args.device)
                    v_predicted = model(vx.flatten(start_dim=1))
                    v_accuracy = np.sum((torch.argmax(v_predicted, dim=-1) == vy).cpu().numpy()) / len(vy)

                    # test data
                    tx, ty = next(iter(test_loader))
                    tx, ty = tx.to(args.device), ty.to(args.device)
                    t_predicted = model(tx.flatten(start_dim=1))
                    t_accuracy = np.sum((torch.argmax(t_predicted, dim=-1) == ty).cpu().numpy()) / len(ty)

                    valid_accs.append(v_accuracy)
                    test_accs.append(t_accuracy)

                    if v_accuracy > max_v:
                        max_v = v_accuracy
                    if t_accuracy > max_t:
                        max_t = t_accuracy

                    # print
                    print('')
                    print(Fore.LIGHTGREEN_EX + f'[valid] Episode {epoch + 1}:\t \033[1mSCORE\033[0m = {v_accuracy:0.3f}'
                          + Fore.LIGHTGREEN_EX + f' \t \033[1mMAX SCORE\033[0m = {max_v:0.3f}')
                    print(Fore.GREEN + f'[test] Episode {epoch + 1}:\t \033[1mSCORE\033[0m = {t_accuracy:0.3f}'
                          + Fore.GREEN + f' \t \033[1mMAX SCORE\033[0m = {max_t:0.3f}')

        train_accs.append(np.mean(epoch_accs))
        print('')
    torch.save({'train_accs': torch.tensor(train_accs),
                'valid_accs': torch.tensor(valid_accs),
                'test_accs': torch.tensor(test_accs)}, log_dir)


if __name__ == "__main__":
    parser = get_arg_parser()
    main(parser.parse_args())
