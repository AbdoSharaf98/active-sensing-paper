import torch
from colorama import Fore
from pytorch_lightning.loggers import CSVLogger
from torch import nn
import numpy as np
from typing import Sequence, NamedTuple

from torch.utils.tensorboard import SummaryWriter

from nets import create_ff_network
from distributions import Gaussian
import os


class ObservationEncoder(nn.Module):

    def __init__(self, obs_dim: int,
                 latent_dim: int,
                 activation: str = 'relu',
                 layers: list[int] = None):
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        if layers is None:
            layers = [256, 256]

        # encoder layers
        self.encoder = create_ff_network([obs_dim] + layers + [latent_dim],
                                         h_activation=activation,
                                         out_activation="tanh")

    def forward(self, x):
        return self.encoder(x)


class ActorCritic(nn.Module):

    def __init__(self, actor_obs_encoder: ObservationEncoder,
                 critic_obs_encoder: ObservationEncoder,
                 action_dim: int,
                 activation: str = 'relu',
                 actor_std: float = 0.05,
                 actor_layers: list[int] = None,
                 critic_layers: list[int] = None,
                 lr: float = 0.001):

        super().__init__()

        self.action_dim = action_dim
        self.actor_std = actor_std
        self.policy_obs_encoder = actor_obs_encoder
        self.critic_obs_encoder = critic_obs_encoder

        # build the actor
        if actor_layers is None:
            actor_layers = [64]
        actor_layers = [actor_obs_encoder.latent_dim] + actor_layers + [action_dim]
        self.actor_mu = create_ff_network(
            layer_dims=actor_layers, h_activation=activation, out_activation="tanh"
        )

        # build the critic
        if critic_layers is None:
            critic_layers = [64]
        critic_layers = [critic_obs_encoder.latent_dim] + critic_layers + [1]
        self.critic = create_ff_network(
            layer_dims=critic_layers, h_activation=activation
        )

        # critic optimizer
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr)

    def forward(self, x):

        # compute action
        z_actor = self.policy_obs_encoder(x)
        action_mean = self.actor_mu(z_actor)
        action_dist = Gaussian(action_mean, self.actor_std * torch.ones_like(action_mean))
        action = action_dist.sample().clamp(-1, 1)
        log_prob = action_dist.log_prob(action)

        # compute critic value
        z_critic = self.critic_obs_encoder(x)
        value = self.critic(z_critic)

        return action, log_prob, value.squeeze(-1)


class WorldModel(nn.Module):

    def __init__(self, latent_dim: int,
                 action_dim: int,
                 activation: str = "relu",
                 layers: list[int] = None):
        super().__init__()

        self.latent_dim, self.action_dim = latent_dim, action_dim

        if layers is None:
            layers = [64, 64]
        input_dim = latent_dim + action_dim
        layers = [input_dim] + layers + [latent_dim]
        self.predictor = create_ff_network(
            layer_dims=layers, h_activation=activation
        )

    def forward(self, z, action):
        inp = torch.cat([z, action], dim=-1)
        return self.predictor(inp)


class BYOLActiveSensor(nn.Module):
    DEFAULT_CONFIG = {
        "lr": 2.5e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.15,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "target_update_rate": 0.001
    }

    def __init__(self, env,
                 online_encoder: ObservationEncoder,
                 target_encoder: ObservationEncoder,
                 actor_critic: ActorCritic,
                 world_model: WorldModel,
                 decider,
                 log_dir: str,
                 config: dict = None,
                 device: str = 'cuda'):
        super().__init__()

        self.config = self.DEFAULT_CONFIG if config is None else config
        self.env = env
        self.device = device

        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.actor_critic = actor_critic
        self.world_model = world_model
        self.decider = decider

        # ema parameters for reward normalization
        self.c = 1  # counter for reward normalization
        self.r_bar, self.r_bar2 = 0.0, 0.0
        self.alpha_r = 0.99
        self.mu_r, self.mu_r2 = 0.0, 0.0
        self.sigma_r = 1e-4

        # create a logger
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)
        self.csv_logger = CSVLogger(save_dir=log_dir, name='')
        self.checkpoint_dir = log_dir

        # create an optimizer for the world model
        self.wm_optimizer = torch.optim.Adam(
            list(self.world_model.parameters()) + list(self.online_encoder.parameters()),
            lr=self.config['lr']
        )

    def _normalize_rewards(self, raw_rewards):
        rc = torch.mean(raw_rewards).item()
        rc2 = torch.mean(raw_rewards ** 2).item()
        self.r_bar = self.alpha_r * self.r_bar + (1 - self.alpha_r) * rc
        self.r_bar2 = self.alpha_r * self.r_bar2 + (1 - self.alpha_r) * rc2
        self.mu_r = self.r_bar / (1 - self.alpha_r ** self.c)
        self.mu_r2 = self.r_bar2 / (1 - self.alpha_r ** self.c)
        self.sigma_r = (max(self.mu_r2 - self.mu_r ** 2, 0) + 1e-8) ** 0.5
        self.c += 1
        return raw_rewards / self.sigma_r

    def _get_advantages(self, gae, next_value, value, reward):

        delta = reward + self.config["gamma"] * next_value - value
        gae = (
                delta + self.config["gamma"] * self.config["gae_lambda"] * gae
        )

        return gae

    def _calculate_gae(self, values, rewards):

        advantages = torch.zeros_like(values)
        advantages[:, -1] = rewards[:, -1] - values[:, -1]

        for t in range(values.shape[-1] - 2, -1, -1):
            advantages[:, t] = self._get_advantages(advantages[:, t + 1],
                                                    values[:, t + 1],
                                                    values[:, t],
                                                    rewards[:, t])

        return advantages

    def _compute_rl_losses(self, states, actions, log_probs, rewards, values, advantages, targets):

        # rerun networks
        flat_states = states.flatten(0, 1)
        re_actions, re_logp, re_values = self.actor_critic(flat_states)
        re_actions = re_actions.reshape(actions.shape[:2] + (-1,))
        re_logp = re_logp.reshape(log_probs.shape)
        re_values = re_values.reshape(values.shape)

        # calculate value loss
        value_pred_clipped = values + torch.clip(
            re_values - values, -self.config["clip_eps"], self.config["clip_eps"]
        )
        value_losses = (re_values - targets) ** 2
        value_losses_clipped = (value_pred_clipped - targets) ** 2
        value_loss = (
                0.5 * torch.maximum(value_losses, value_losses_clipped).mean()
        )

        # calculate actor loss
        # # note we skip the first action here bc it's random and has no real log prob
        ratio = torch.exp(re_logp[:, :-1] - log_probs[:, 1:])
        gae = (advantages[:, 1:] - advantages[:, 1:].mean(-1, keepdim=True)) / (advantages[:, 1:].std(-1, keepdim=True) + 1e-8)
        actor_loss1 = ratio * gae
        actor_loss2 = (
                torch.clip(
                    ratio,
                    1.0 - self.config["clip_eps"],
                    1.0 + self.config["clip_eps"]
                ) * gae
        )
        actor_loss = -torch.minimum(actor_loss1, actor_loss2).mean()

        # TODO: add entropy loss

        total_loss = (
                actor_loss
                + self.config["vf_coef"] * value_loss
        )

        return total_loss, value_loss, actor_loss

    def _compute_wm_loss(self, states, actions):

        # calculate world model loss
        z_t_m = self.online_encoder(states[:, :-1].flatten(0, 1))
        z_t = self.target_encoder(states[:, 1:].flatten(0, 1))
        a_t = actions[:, 1:].flatten(0, 1)
        z_pred = self.world_model(z_t_m, a_t.detach())
        wm_loss = nn.functional.mse_loss(z_pred, z_t)

        return wm_loss

    def _update_target_ema(self):
        # TODO: calculate and return the distance between the two parameter sets
        with torch.no_grad():  # no gradients are computed during the update
            tau = 1 - self.config["target_update_rate"]
            for target_param, online_param in zip(self.target_encoder.parameters(),
                                                  self.online_encoder.parameters()):
                # Update the target network's parameters using the formula:
                # θ_target = τ * θ_target + (1 - τ) * θ_online
                target_param.data.copy_(tau * target_param.data + (1.0 - tau) * online_param.data)

    def save_checkpoint(self, save_name: str, loss_dict: dict):

        chkpt_dict = {
            'world_model': self.world_model.state_dict(),
            'online_encoder': self.online_encoder.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'decider': self.decider.state_dict(),
            ''
            'losses': loss_dict
        }

        torch.save(chkpt_dict, os.path.join(self.checkpoint_dir, save_name))

        return chkpt_dict

    def decide(self, states, actions):

        decision_dist = self.decider(torch.cat([states, actions], dim=-1))

        # get the decision
        decision = torch.argmax(decision_dist, dim=-1)

        return decision_dist, decision

    def sensing_loop(self, validation: bool = False,
                     testing: bool = False,
                     with_batch: tuple[torch.Tensor, torch.Tensor] = None,
                     random_action: bool = False):
        """
        :param validation: whether to use a validation batch from the environment
        :param with_batch: specific batch to run the sensing loop with
        :param testing:
        :param random_action:
        :return:
        """

        # reset the environment
        state, action = self.env.reset(validation=validation, testing=testing, with_batch=with_batch)
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)

        # construct tensor to hold episode data
        bsz = len(self.env.current_batch[0])
        states = torch.zeros((bsz, self.env.n_samples + 1, self.env.observation_space.shape[-1])).to(self.device)
        actions = torch.zeros((bsz, self.env.n_samples + 1, self.env.action_space.shape[-1])).to(self.device)
        log_probs = torch.zeros((bsz, self.env.n_samples + 1)).to(self.device)
        rewards = torch.zeros((bsz, self.env.n_samples + 1)).to(self.device)
        values = torch.zeros((bsz, self.env.n_samples + 1)).to(self.device)

        states[:, 0, :] = state  # initial state
        actions[:, 0, :] = action  # initial action

        # go through the allowed number of steps
        for t in range(self.env.n_samples):
            # select action
            if not random_action:
                action, log_prob, value = self.actor_critic(state)
            else:
                action = torch.FloatTensor(bsz, self.world_model.action_dim).uniform_(-1, 1).to(self.device)
                log_prob, value = 0.0, 0.0
            # step the environment
            next_state = self.env.step(action.detach().cpu().numpy())[0]
            next_state = torch.Tensor(next_state).to(self.device)

            # calculate the distance between predicted and actual observation
            # # prediction
            z_t_m1 = self.online_encoder(state)
            z_hat_t = self.world_model(z_t_m1, action)

            # # ground truth
            z_t = self.target_encoder(next_state)

            # # set the reward to the distance between prediction and ground truth
            reward = torch.linalg.norm(z_t - z_hat_t, dim=-1, ord=2)
            state = next_state

            # store
            states[:, t + 1, :] = state
            actions[:, t + 1, :] = action
            log_probs[:, t + 1] = log_prob
            rewards[:, t + 1] = reward
            values[:, t] = value

        # calculate the last value
        _, _, values[:, -1] = self.actor_critic(state)

        # normalize rewards
        rewards = self._normalize_rewards(rewards)

        return states, actions, log_probs, rewards, values,

    def training_step(self, update=True, random_action=False):

        states, actions, log_probs, rewards, values = self.sensing_loop(validation=False,
                                                                        random_action=random_action)

        # compute advantages
        advantages = self._calculate_gae(values, rewards)
        targets = advantages + values

        # compute decision and decision loss
        decision_dist, decision = self.decide(states, actions.detach())
        _, accuracy, _, info = self.env.step(decision.squeeze().cpu().numpy())
        labels = info['true_labels'].to(self.device)
        decision_loss = nn.functional.cross_entropy(decision_dist.squeeze(), labels)

        # compute losses
        # # RL losses
        total_loss, value_loss, actor_loss = self._compute_rl_losses(states, actions, log_probs, rewards, values,
                                                                     advantages, targets)
        # # world model loss
        wm_loss = self._compute_wm_loss(states, actions)

        # update the policy
        if (not random_action) and update:
            self.actor_critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor_critic.optimizer.step()

        # update the world model
        if update:
            self.wm_optimizer.zero_grad()
            wm_loss.backward()
            self.wm_optimizer.step()

            if not random_action:
                self.decider.optimizer.zero_grad()
                decision_loss.backward()
                self.decider.optimizer.step()

            # update the target model using moving averages
            self._update_target_ema()

        return accuracy, decision_loss, actor_loss, value_loss, wm_loss

    def validation_step(self, entropy_thresh=None, testing=False):
        # NOTE: if the validation batch is too large & we are running on cuda, it will run out of memory
        num_valid = len(self.env.valid_loader if not testing else self.env.test_loader)
        val_accs = torch.zeros((num_valid,))
        val_losses = torch.zeros((num_valid,))
        val_steps = torch.zeros((num_valid,))
        for v in range(num_valid):
            # sense
            val_states, val_actions = self.sensing_loop(validation=not testing, testing=testing)[:2]
            # decide
            val_decision_dist, val_decision = self.decide(val_states, val_actions)
            # get validation accuracy
            _, val_accuracy, _, val_info = self.env.step(val_decision.squeeze().cpu().numpy())
            val_labels = val_info['true_labels'].to(self.device)

            # compute the validation decision loss
            val_decision_loss = torch.nn.functional.cross_entropy(val_decision_dist.squeeze(), val_labels)

            val_accs[v] = val_accuracy
            val_losses[v] = val_decision_loss

        return val_accs.mean(), val_losses.mean()

    def learn(self,
              num_epochs: int,
              log_every: float = 5,
              validate_every: float = 3,
              start_epoch: int = 0,
              num_random_epochs: int = 0):
        """
        :param start_epoch:
        :param num_random_epochs:
        :param num_epochs: number of epochs to train for
        :param log_every: log every x epochs (if fraction then logging after a given number of batches)
        :param validate_every: validate every x epochs (if fraction then validate after a given number of batches)
        :return:
        """

        # get the number of updates to log/validate after
        num_updates_per_epoch = len(self.env.train_loader)
        log_interval = int(log_every * num_updates_per_epoch)
        validate_interval = int(validate_every * num_updates_per_epoch)

        # initialize array to track training and validation accuracies
        max_val_acc = 0
        max_tst_acc = 0

        # no of training batches
        num_train = len(self.env.train_loader)

        # initial checkpoint
        best_model_checkpoint = {}

        # training loop
        total_updates = start_epoch * num_updates_per_epoch
        for epoch in range(start_epoch, num_epochs):

            # will be used to calculate avg training accuracy at the end of each epoch
            train_accs = torch.zeros((num_train,))

            rnd_act = epoch < num_random_epochs

            for batch_num in range(num_train):

                accuracy, decision_loss, actor_loss, value_loss, wm_loss = self.training_step(update=True,
                                                                                              random_action=rnd_act)

                # step the number of total updates
                if not rnd_act:
                    total_updates += 1

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
                    self.logger.add_scalar('train/decision_loss', decision_loss, total_updates)
                    self.logger.add_scalar('train/actor_loss', actor_loss, total_updates)
                    self.logger.add_scalar('train/value_loss', value_loss, total_updates)
                    self.logger.add_scalar('train/wm_loss', wm_loss, total_updates)

                # validate
                if (total_updates % validate_interval == 0) and (not rnd_act):
                    avg_val_accuracy, avg_val_loss = self.validation_step(testing=False)
                    avg_tst_accuracy, avg_tst_loss = self.validation_step(testing=True)

                    # checkpoint if max val acc has been exceeded
                    if avg_val_accuracy >= max_val_acc:
                        max_val_acc = avg_val_accuracy
                        best_model_checkpoint = self.save_checkpoint(
                            f'last',
                            {'accuracy': avg_val_accuracy, 'decision_loss': avg_val_loss})
                    max_tst_acc = max(max_tst_acc, avg_tst_accuracy)
                    print('')
                    print(
                        Fore.LIGHTGREEN_EX + f'[valid] Episode {epoch + 1}:\t \033[1mSCORE\033[0m = {avg_val_accuracy:0.3f}'
                        + Fore.LIGHTGREEN_EX + f' \t \033[1mMAX SCORE\033[0m = {max_val_acc:0.3f}')
                    print(Fore.GREEN + f'[test] Episode {epoch + 1}:\t \033[1mSCORE\033[0m = {avg_tst_accuracy:0.3f}'
                          + Fore.GREEN + f' \t \033[1mMAX SCORE\033[0m = {max_tst_acc:0.3f}')

                    # log
                    self.logger.add_scalar('valid/accuracy', avg_val_accuracy, total_updates)
                    self.logger.add_scalar('valid/decision_loss', avg_val_loss, total_updates)
                    self.logger.add_scalar('test/accuracy', avg_tst_accuracy, total_updates)
                    self.logger.add_scalar('test/decision_loss', avg_tst_loss, total_updates)

                    self.csv_logger.log_metrics({'test_acc': avg_tst_accuracy}, step=epoch)
                    self.csv_logger.save()

            # print a new line for the next epoch
            print('')
