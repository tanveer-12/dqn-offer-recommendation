"""DQN Agent implementation for customer offer recommendation"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent:
    """Interacts with and learns from the environment using DQN"""

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 seed: int,
                 buffer_size: int = int(1e5),
                 batch_size: int = 64,
                 gamma: float = 0.99,   # discount factor
                 lr: float = 5e-4,
                 update_every: int = 4,
                 tau: float = 1e-3):    # for soft update of target network
        """
        
        """
        