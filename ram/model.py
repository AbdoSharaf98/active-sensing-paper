# *********************************************************************** #
# Adapted from https://github.com/kevinzakka/recurrent-visual-attention
# *********************************************************************** #

import torch.nn as nn
from ram import modules
import torch
import torch.nn.functional as F
from envs.entry_points.active_sensing import ActiveSensingEnv
import os
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore


class RecurrentAttentionModel(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].
    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.
    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
            self, env, h_g, h_l, std, hidden_size,
            log_dir,
            random_strategy,
            checkpoint_dir=None,
            lr=0.001,
            device='cuda',
            discrete_loc=False
    ):
        """Constructor.
        Args:
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
        """
        super().__init__()

        self.std = std
        self.lr = lr
        self.device = device

        self.env = env
        self.num_glimpses = env.n_samples

        # model parameters
        s = env.fovea_scale
        c = env.num_channels
        k = env.num_foveated_patches
        g = env.sample_dim

        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.rnn = modules.CoreNetwork(h_g + h_l, hidden_size)
        if not discrete_loc:
            self.locator = modules.LocationNetwork(hidden_size, 2, std)
        else:
            self.locator = modules.DiscreteLocationNetwork(hidden_size, random_strategy.action_grid)
        self.classifier = modules.ActionNetwork(hidden_size, env.num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # construct a logger
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

        # checkpointing dir
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else log_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.random_strategy = random_strategy

        self.to(self.device)

    def save_checkpoint(self, save_name: str, loss_dict: dict):

        # TODO: we may want to save the perception model's hyper-parameters

        chkpt_dict = {
            'state_dict': self.state_dict(),
            'losses': loss_dict
        }

        torch.save(chkpt_dict, os.path.join(self.checkpoint_dir, save_name))

        return chkpt_dict

    def load_from_checkpoint_dict(self, checkpoint_dict):

        self.load_state_dict(checkpoint_dict['state_dict'])

    def forward(self, x, l_t_prev, h_t_prev, last=False, random_action=False):
        """Run RAM for one timestep on a minibatch of images.
        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.
        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t, h_t_prev)

        if not random_action:
            log_pi, l_t = self.locator(h_t)
        else:
            l_t = self.random_strategy.select_action(x)
            log_pi = (1/self.random_strategy.action_grid.num_actions) * torch.ones_like(l_t).float().to(self.device)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi

    def reset(self, batch_size):
        h_t = torch.zeros(
            batch_size,
            self.rnn.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True
        )
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def training_step(self, batch, random_action=False):
        x, y = batch

        self.optimizer.zero_grad()

        x, y = x.to(self.device), y.to(self.device)

        # initial location and hidden state
        h_t, l_t = self.reset(batch_size=x.shape[0])

        # extract glimpses
        locs = []
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses - 1):
            # forward pass the model
            h_t, l_t, b_t, p = self.forward(x, l_t, h_t, random_action=random_action)

            """
            if random_action:
                l_t = torch.FloatTensor(x.shape[0], 2).uniform_(-1, 1).to(self.device)
                p = 0.25 * torch.ones_like(l_t).float().to(self.device)
            """

            # store
            locs.append(l_t[0:9])
            baselines.append(b_t)
            log_pi.append(p)

        # last iteration
        h_t, l_t, b_t, log_probas, p = self.forward(x, l_t, h_t, last=True, random_action=random_action)
        """
        if random_action:
            l_t = torch.FloatTensor(x.shape[0], 2).uniform_(-1, 1).to(self.device)
            p = 0.25 * torch.ones_like(l_t).float().to(self.device)
        """
        log_pi.append(p)
        baselines.append(b_t)
        locs.append(l_t[0:9])

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)

        # calculate reward
        predicted = torch.max(log_probas, 1)[1]
        R = (predicted.detach() == y).float()
        R = R.unsqueeze(1).repeat(1, self.num_glimpses)

        # compute losses for differentiable modules
        loss_action = F.nll_loss(log_probas, y)
        loss_baseline = F.mse_loss(baselines, R) if not random_action else 0

        # compute reinforce loss
        # summed over timesteps and averaged across batch
        if not random_action:
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)
        else:
            loss_reinforce = 0

        # sum up into a hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce * 0.01

        # compute accuracy
        correct = (predicted == y).float()
        acc = (correct.sum() / len(y))

        # compute gradients and update SGD
        loss.backward()
        self.optimizer.step()

        return acc, loss

    def validation_step(self, M=1):

        avg_loss = 0
        avg_acc = 0

        for i, (x, y) in enumerate(self.env.valid_loader):

            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(M, 1, 1, 1)

            # initialize location vector and hidden state
            h_t, l_t = self.reset(x.shape[0])

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.forward(x, l_t, h_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.forward(x, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # compute accuracy
            correct = (predicted == y).float()
            acc = correct.sum() / len(y)

            # store
            avg_loss += loss.item() / len(self.env.valid_loader)
            avg_acc += acc.item() / len(self.env.valid_loader)

        return avg_acc, avg_loss

    def learn(self, num_epochs: int,
              M: int = 1,
              log_every: int = 5,
              validate_every: int = 3,
              num_random_epochs: int = 1):

        # get the number of updates to log/validate after
        num_updates_per_epoch = len(self.env.train_loader)
        log_interval = int(log_every * num_updates_per_epoch)
        validate_interval = int(validate_every * num_updates_per_epoch)

        # initialize array to track training and validation accuracies
        max_val_acc = 0

        # no of training batches
        num_train = len(self.env.train_loader)

        # TODO: add learning rate schedulers

        # initial checkpoint
        best_model_checkpoint = {}

        # training loop
        total_updates = 0
        for epoch in range(num_epochs):

            # will be used to calculate avg training accuracy at the end of each epoch
            train_accs = torch.zeros((num_train,))

            for batch_num, batch in enumerate(self.env.train_loader):

                accuracy, loss = self.training_step(batch, random_action=epoch < num_random_epochs)

                # store accuracy to calculate running average
                train_accs[batch_num] = accuracy

                # print progress
                print(Fore.YELLOW + f'[train] Episode {epoch + 1} ({batch_num + 1}/{num_train}):\t \033[1mSCORE\033['
                                    f'0m = {accuracy:0.3f}'
                      + Fore.YELLOW + f' \t \033[1mAVG SCORE\033[0m = {train_accs[:batch_num + 1].mean():0.3f}',
                      end='\r')

                # log
                if total_updates % log_interval == 0:
                    self.logger.add_scalar('train/accuracy', accuracy, total_updates)
                    self.logger.add_scalar('train/loss', loss, total_updates)

                # validate
                if (total_updates % validate_interval == 0) and (epoch >= num_random_epochs):
                    avg_val_accuracy, avg_val_loss = self.validation_step(M=M)

                    # checkpoint if max val acc has been exceeded
                    if avg_val_accuracy >= max_val_acc:
                        max_val_acc = avg_val_accuracy
                        best_model_checkpoint = self.save_checkpoint(
                            f'bestModel_epoch={total_updates / num_updates_per_epoch:0.2f}',
                            {'accuracy': avg_val_accuracy, 'loss': avg_val_loss})

                    print(Fore.GREEN + f'[valid] Episode {epoch + 1}:\t \033[1mSCORE\033[0m = {avg_val_accuracy:0.3f}'
                          + Fore.GREEN + f' \t \033[1mMAX SCORE\033[0m = {max_val_acc:0.3f}')

                    # log
                    self.logger.add_scalar('valid/accuracy', avg_val_accuracy, total_updates)
                    self.logger.add_scalar('valid/loss', avg_val_loss, total_updates)

                # step the number of total updates
                total_updates += 1
            # print a new line for the next epoch
            print('')

        # # after training is done, use the best model to perform a train and validation pass
        # load the best state dict
        self.load_from_checkpoint_dict(best_model_checkpoint)

        # pass through the training data
        best_train_avg_score, best_train_avg_loss, best_train_avg_steps = 0, 0, 0
        for _, batch in enumerate(self.env.train_loader):
            accuracy, loss = self.training_step(batch)
            best_train_avg_score += accuracy / num_train
            best_train_avg_loss += loss / num_train

        # pass through the validation data
        best_valid_avg_score, best_valid_avg_loss = self.validation_step(M=M)

        # print
        print(Fore.GREEN + f'[BEST]:\t \033[1mTRAIN SCORE\033[0m = {best_train_avg_score:0.3f}'
              + Fore.GREEN + f' \t \033[1mVALID SCORE\033[0m = {best_valid_avg_score:0.3f}')
