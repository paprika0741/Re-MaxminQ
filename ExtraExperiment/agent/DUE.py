from agent. DQN import DQN
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import torch.optim as optim
from utils. networks import ANetwork
from utils. memory import ReplayMemory


class DUE(DQN):
    '''
    Implementation of Double DQN with target network and replay buffer
    '''
    def __init__(
        self,
        action_dim: int,
        device,
        gamma: float = 0.99,
        seed: int = 0,
        eps_start: float = 1.,
        eps_final: float = 0.1,
        eps_decay: float = 1000_000,
        lr: float = 0.0000625,
        batch_size: int = 16,
        memory_size: int = 100_000,
        stack_size: int = 4,
        load_model: bool = False,
        model_index: int = 0,
    ):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.seed = seed
        #epsilon for epsilon greedy policy
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.eps = eps_start

        self.r = random.Random()
        self.r.seed(seed)
        # init network
        self.Q_net = ANetwork(action_dim).to(device)
        self.Q_target = ANetwork(action_dim).to(device)
        # init network parameters
        self.Q_net.apply(ANetwork.init_weights)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        #
        self.Q_target.eval()

        self.optimizer = optim.Adam(
            self.Q_net.parameters(),
            lr=lr,
            eps=1.5e-4,
        )

        self.batch_size = batch_size
        #replay buffer
        self.memory = ReplayMemory(stack_size + 1, memory_size, device)


