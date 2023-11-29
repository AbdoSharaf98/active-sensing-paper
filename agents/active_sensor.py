import os
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from envs.entry_points.active_sensing import ActiveSensingEnv
from models.action import ActionStrategy, ActionNetworkStrategy, DirectEvaluationStrategy, RandomActionStrategy
from models.perception import PerceptionModel
from nets import DecisionNetwork
from colorama import Fore
from utils.training import PerceptionModelTrainer


class BayesianActiveSensor(nn.Module):
    """ Full active sensor model """

    def __init__(self, env: ActiveSensingEnv,
                 perception_model: PerceptionModel,
                 actor: ActionStrategy,
                 decider: DecisionNetwork,
                 log_dir: str,
                 checkpoint_dir: str = None,
                 device: str = 'cuda',
                 decider_input: str = 'perception'):
        """

        :param env: active sensing environment the agent is in
        :param perception_model: the perception model component
        :param actor: the action selection strategy
        :param decider: the decision network component
        :param log_dir: directory to log training results
        :param checkpoint_dir: directory to checkpoint model during training
        :param device: device on which model components live
        :param decider_input: whether the decider takes perception states as input or takes raw collected observations
                              options are: 'perception', 'raw'
        """

        super().__init__()

        self.env = env
        self.perception_model = perception_model
        self.actor = actor
        self.decider = decider
        self.device = device
        self.decider_input = decider_input

        # create a logger
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

        # checkpointing dir
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else log_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # in case we need to warm-up with random actions
        self._random_strategy = None

    @property
    def random_strategy(self):
        if self._random_strategy is None:
            return RandomActionStrategy(self.perception_model, action_grid_size=self.actor.action_grid.grid_size)
        return self._random_strategy

    def save_checkpoint(self, save_name: str, loss_dict: dict):

        # TODO: we may want to save the perception model's hyper-parameters

        chkpt_dict = {
            'perception_model': self.perception_model.state_dict(),
            'actor': self.actor.state_dict(),
            'decider': self.decider.state_dict(),
            'losses': loss_dict
        }

        torch.save(chkpt_dict, os.path.join(self.checkpoint_dir, save_name))

        return chkpt_dict

    def load_from_checkpoint_dict(self, chkpt_dict: dict):

        self.perception_model.load_state_dict(chkpt_dict['perception_model'])
        if isinstance(self.actor, ActionNetworkStrategy):
            self.actor.action_net.load_state_dict(chkpt_dict['actor'])
        self.decider.load_state_dict(chkpt_dict['decider'])

    def sensing_loop(self, validation: bool = False,
                     testing: bool = False,
                     with_batch: tuple[torch.Tensor, torch.Tensor] = None,
                     entropy_thresh: float = None,
                     random_action: bool = False):
        """
        :param entropy_thresh: threshold on the uncertainty used to stop sensing
        :param validation: whether to use a validation batch from the environment
        :param with_batch: specific batch to run the sensing loop with
        :param testing:
        :param random_action:
        :return:
        """

        # reset the perception model
        self.perception_model.reset_rnn_states()

        # reset the environment
        state, action = self.env.reset(validation=validation, testing=testing, with_batch=with_batch)

        # construct tensor to hold episode data
        bsz = len(self.env.current_batch[0])
        states = torch.zeros((bsz, self.env.n_samples + 1, self.env.observation_space.shape[-1])).to(self.device)
        actions = torch.zeros((bsz, self.env.n_samples + 1, self.env.action_space.shape[-1])).to(self.device)

        states[:, 0, :] = torch.tensor(state).to(self.device)  # initial state
        actions[:, 0, :] = torch.tensor(action).to(self.device)  # initial action

        # put the actor in training mode based on whether we are validating
        self.actor.train((not validation) and (not testing))

        # go through the allowed number of steps
        for t in range(self.env.n_samples):
            # select action
            if not random_action:
                action = self.actor.select_action(states[:, :t + 1, :].detach(), actions[:, :t + 1, :].detach())
            else:
                action = self.random_strategy.select_action(states)

            # step the environment
            state = self.env.step(action.detach().cpu().numpy())[0]

            # store
            states[:, t + 1, :] = torch.tensor(state).to(self.device)
            actions[:, t + 1, :] = action

            # decide whether to stop based on the agent's uncertainty
            if entropy_thresh is not None:
                uncertainty = self.perception_model(states[:, :t + 2, :],
                                                    actions[:, :t + 2, :])[-1].torch_dist.entropy()
                if torch.max(uncertainty).item() <= entropy_thresh:
                    self.env.relinquish_views()
                    break

        states = states[:, :t + 2, :]
        actions = actions[:, :t + 2, :]

        return states, actions

    def decide(self, states, actions):

        if self.decider_input == 'perception':
            s_dist = self.perception_model(states, actions)[-1]
            decision_dist = self.decider(s_dist.mu)
        else:
            decision_dist = self.decider(torch.cat([states, actions], dim=-1))

        # get the decision
        decision = torch.argmax(decision_dist, dim=-1)

        return decision_dist, decision

    def training_step(self, entropy_thresh=None, beta=0.1, update=True, random_action=False):

        states, actions = self.sensing_loop(validation=False, entropy_thresh=entropy_thresh,
                                            random_action=random_action)

        # make a decision
        decision_dist, decision = self.decide(states, actions)

        # make the decision and get feedback from the environment
        _, accuracy, dones, info = self.env.step(decision.squeeze().cpu().numpy())
        labels = info['true_labels'].to(self.device)

        # compute the decision loss
        decision_loss = torch.nn.functional.cross_entropy(decision_dist.squeeze(), labels)

        # compute the perception model losses
        p_loss, rec_loss, z_kl_loss, s_kl_loss = self.perception_model._compute_losses(states,
                                                                                       actions,
                                                                                       beta=beta)
        if update:

            # optimize the decider
            if not random_action:
                self.decider.optimizer.zero_grad()
                decision_loss.backward()
                self.decider.optimizer.step()

            # optimize the perception model
            self.perception_model.manual_optimizer.zero_grad()
            p_loss.backward()
            # clip gradients to prevent gradient explosion
            # nn.utils.clip_grad_norm_(self.perception_model.parameter_list(), 10)
            self.perception_model.manual_optimizer.step()

        return accuracy, decision_loss, (p_loss, rec_loss, z_kl_loss, s_kl_loss), states.shape[1] - 1

    def validation_step(self, entropy_thresh=None, testing=False):
        # NOTE: if the validation batch is too large & we are running on cuda, it will run out of memory
        num_valid = len(self.env.valid_loader if not testing else self.env.test_loader)
        val_accs = torch.zeros((num_valid,))
        val_losses = torch.zeros((num_valid,))
        val_steps = torch.zeros((num_valid,))
        for v in range(num_valid):
            # sense
            val_states, val_actions = self.sensing_loop(validation=not testing,
                                                        testing=testing,
                                                        entropy_thresh=entropy_thresh)
            # decide
            val_decision_dist, val_decision = self.decide(val_states, val_actions)
            # get validation accuracy
            _, val_accuracy, _, val_info = self.env.step(val_decision.squeeze().cpu().numpy())
            val_labels = val_info['true_labels'].to(self.device)

            # compute the validation decision loss
            val_decision_loss = torch.nn.functional.cross_entropy(val_decision_dist.squeeze(), val_labels)

            val_accs[v] = val_accuracy
            val_losses[v] = val_decision_loss
            val_steps[v] = val_states.shape[1] - 1

        return val_accs.mean(), val_losses.mean(), val_steps.mean()

    def learn(self,
              num_epochs: int,
              entropy_thresh: float = None,
              log_every: float = 5,
              validate_every: float = 3,
              beta_sched: np.ndarray = None,
              num_random_epochs: int = 0):
        """
        :param num_random_epochs:
        :param beta_sched:
        :param num_epochs: number of epochs to train for
        :param entropy_thresh: entropy threshold to control when the agent stops sensing before using max samples
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

        # define the beta schedule if it's not given
        if beta_sched is None:
            beta_schedule = np.ones((num_epochs,))

        # TODO: add learning rate schedulers

        # initial checkpoint
        best_model_checkpoint = {}

        # training loop
        total_updates = 0
        for epoch in range(num_epochs):

            # will be used to calculate avg training accuracy at the end of each epoch
            train_accs = torch.zeros((num_train,))

            rnd_act = epoch < num_random_epochs

            for batch_num in range(num_train):

                accuracy, decision_loss, perception_losses, steps = self.training_step(entropy_thresh,
                                                                                       beta_sched[epoch],
                                                                                       random_action=rnd_act)

                # step the number of total updates
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
                    self.logger.add_scalar('train/rec_loss', perception_losses[1], total_updates)
                    self.logger.add_scalar('train/z_kl_loss', perception_losses[2], total_updates)
                    self.logger.add_scalar('train/s_kl_loss', perception_losses[3], total_updates)
                    self.logger.add_scalar('train/steps', steps, total_updates)

                # validate
                if (total_updates % validate_interval == 0) and (not rnd_act):
                    avg_val_accuracy, avg_val_loss, avg_val_steps = self.validation_step(testing=False,
                                                                                         entropy_thresh=entropy_thresh)
                    avg_tst_accuracy, avg_tst_loss, avg_tst_steps = self.validation_step(testing=True,
                                                                                         entropy_thresh=entropy_thresh)

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
                    self.logger.add_scalar('valid/steps', avg_val_steps, total_updates)
                    self.logger.add_scalar('test/accuracy', avg_tst_accuracy, total_updates)
                    self.logger.add_scalar('test/decision_loss', avg_tst_loss, total_updates)
                    self.logger.add_scalar('test/steps', avg_tst_steps, total_updates)

            # print a new line for the next epoch
            print('')

        # # after training is done, use the best model to perform a train and validation pass
        # load the best state dict
        self.load_from_checkpoint_dict(best_model_checkpoint)

        # pass through the training data
        best_train_avg_score, best_train_avg_loss, best_train_avg_steps = 0, 0, 0
        for batch_num in range(num_train):
            accuracy, decision_loss, _, steps = self.training_step(entropy_thresh, update=False)
            best_train_avg_score += accuracy / num_train
            best_train_avg_loss += decision_loss / num_train
            best_train_avg_steps += steps / num_train

        # pass through the validation data
        best_valid_avg_score, best_valid_avg_loss, best_valid_avg_steps = self.validation_step(entropy_thresh)

        # print
        print(Fore.GREEN + f'[BEST]:\t \033[1mTRAIN SCORE\033[0m = {best_train_avg_score:0.3f}'
              + Fore.GREEN + f'\t \033[1mTRAIN STEPS\033[0m = {best_train_avg_steps:0.3f}'
              + Fore.GREEN + f' \t \033[1mVALID SCORE\033[0m = {best_valid_avg_score:0.3f}'
              + Fore.GREEN + f'\t \033[1mVALID STEPS\033[0m = {best_valid_avg_steps:0.3f}')
