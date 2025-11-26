"""Deep Q-network for Customer offer recommendation"""

import torch    # for neural network
import torch.nn as nn   # for neural network layers
import torch.nn.functional as F     # for activation function

class QNetwork(nn.Module):
    """
    Neural network that approximates the Q-values for state-action pairs
    """
    def __init__(self, state_size:int, action_size:int, hidden_units: list = [64, 64]):
        """
        Initialize the neural network

        Args:
            state_size: Number of state features (5 in our case: [loyality, spend, last_offer, accepted, recency])
            action_size: Number of possible actions (5: no offer, 5% disc, 10% disc, BOGO, premium membership)
            hidden_units: List of hidden layer sizes for the network
        """
        # calling parent class (nn.Module) constructor
        super(QNetwork, self).__init__()

        # need layers that connect = state size, hidden units[0], hidden units[1] - action size 
        # using nn.Linear for each layer
        # using nn.ReLU between layers for activation
        self.fc1 = nn.Linear(state_size, 5)
        x = nn.ReLU(self.fc1(x))
        self.fc2 = nn.Linear(state_size, 5)
        


