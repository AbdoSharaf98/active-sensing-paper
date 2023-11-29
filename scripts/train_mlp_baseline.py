import os.path

from colorama import Fore

from utils.data import get_mnist_data, create_data_loaders
from nets import create_ff_network
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get the dataset and create the data loaders
dataset = get_mnist_data(data_version='translated')

seeds = [1211, 1213, 1214, 1215]
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, valid_loader, test_loader = create_data_loaders(dataset, batch_size=64, num_workers=0)

    # create the model
    in_dim, out_dim = 60**2, 10
    layers = [in_dim] + [128, 128] + [out_dim]
    model = create_ff_network(layers, h_activation='relu', out_activation='softmax').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train

    save_dir = f"../runs/translated_mnist/mlp_baseline"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_accs = []
    valid_accs = []
    test_accs = []

    validate_every = 25/len(train_loader)

    num_epochs = 1
    num_train = len(train_loader)
    validate_interval = int(validate_every * num_train)

    max_v = 0.0
    max_t = 0.0
    total_updates = 0
    for epoch in range(num_epochs):
        epoch_accs = np.zeros((num_train,))
        for batch_num, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

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
            accuracy = np.sum((torch.argmax(predicted, dim=-1) == y).cpu().numpy())/len(y)
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
                    vx, vy = vx.to(device), vy.to(device)
                    v_predicted = model(vx.flatten(start_dim=1))
                    v_accuracy = np.sum((torch.argmax(v_predicted, dim=-1) == vy).cpu().numpy())/len(vy)

                    # test data
                    tx, ty = next(iter(test_loader))
                    tx, ty = tx.to(device), ty.to(device)
                    t_predicted = model(tx.flatten(start_dim=1))
                    t_accuracy = np.sum((torch.argmax(t_predicted, dim=-1) == ty).cpu().numpy())/len(ty)

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
                'test_accs': torch.tensor(test_accs)}, os.path.join(save_dir, f'data_{seed}'))

print("DONE")

