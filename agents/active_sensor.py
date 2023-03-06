from distributions import Gaussian
from envs.active_sensing import mnist_active_sensing
from envs.entry_points.active_sensing import ActiveSensingEnv
from copy import deepcopy
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch import nn
from torch.distributions.kl import kl_divergence
from models.perception import PerceptionModel
from models.actors import ActionStrategy
from models.vae_1d import create_ff_network
from nets import DecisionNetwork
from utils.data import get_mnist_data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class ActiveSensor(nn.Module):
    """ Full active sensor model """

    def __init__(self, env: ActiveSensingEnv,
                 perception_model: PerceptionModel,
                 actor: ActionStrategy,
                 decider: DecisionNetwork,
                 log_dir: str,
                 checkpoint_dir: str):
        
        super().__init__()

        self.env = env
        self.perception_model = perception_model
        self.actor = actor
        self.decider = decider

        # create a logger
        self.logger = SummaryWriter(log_dir=log_dir)

        # checkpointing
        # TODO


