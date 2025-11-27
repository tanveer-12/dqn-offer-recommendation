"""Deep Q-Network for customer offer recommendations"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Neural network that approximates Q-values for state-action pairs."""

    def __init__(self,state_size:int, action_size:int, hidden_units: list =[64,64]):
        """
        Initialize the neural network

        WHY?
        - Our customer state is continuous (loyality=0.7, spend=85.5, etc.)
        - We can't use a table for all possible states (infinite!)
        - Neural network generalizes across similar customer states
        - Outputs Q-values for each of the 5 possible offers

        Args:
        state_size: number of state features (5: [loyalty, spend, last_offer, accepted, recency])
        action_size: number of possible actions (5: no offer, 5% disc, 10% dissc, BOGO, premium)
        hidden_units: List of hidden layer sizes [64, 64] means 2 layers of 64 neurons each
        """
        super(QNetwork, self).__init__()
        # Printing to verify if the network is created correctly
        print(f"Initializing QNetwork: state_size={state_size}, action_size={action_size}")

        # Create a list to hold all network layers
        layers = []

        # First layer: Input state -> First hidden layer
        # why: connects 5 customer features to 64 hidden neurons
        layers.append(nn.Linear(state_size, hidden_units[0]))
        # activation: introduces non-linearity so network can learn complex features
        layers.append(nn.ReLU())

        #hidden layers: connect intermediate layers
        # why: if hidden units = [64,64], we connect 64 -> 64 neurons
        for i in range(len(hidden_units) - 1):
            # connect current hidden layer to next hidden layer
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            # add activation after each connection
            layers.append(nn.ReLU())

        # Final layer: last hidden layer -> Output Q-values for each action
        # Why: Maps 64 hidden features to 5 Q-values (one per possible offer)
        layers.append(nn.Linear(hidden_units[-1], action_size))
        # No activation after final layer: We want raw Q-values, not probabilities
        # Raw values allow the agent to learn negative Q-values if needed
        
        # Create sequential network that processes input through all layers
        # nn.Sequential automatically applies each layer in order
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass: Process customer state and return Q-values for all actions.
        Why this method exists:
        - When we call network(state), this method runs automatically
        - Takes customer state as input and outputs predicted Q-values
        - Q-values represent "expected future revenue" for each possible offer
        
        Args:
            state: Current customer state vector [loyalty, spend, last_offer, accepted, recency]
            
        Returns:
            Q-values for each of the 5 possible actions (higher = better expected revenue)
        """
        # Pass the state through all layers and return the result
        # Each layer applies its transformation: Linear -> ReLU -> Linear -> ReLU -> Linear
        return self.network(state)
    
# test the network (runs only if this file is executed directly)
if __name__ == "__main__":
    # create a test network with your parameters
    # state_size=5 (customer features), action_size=5 (offer types)
    test_network = QNetwork(state_size=5, action_size=5)

    # Create a sample customer state (batch size of 1)
    # torch.randn creates random numbers with normal distribution
    sample_state = torch.randn(1, 5)  # 1 customer, 5 features
    q_values = test_network(sample_state)
    print(f"Sample state: {sample_state.flatten()}")
    print(f"Q-values for 5 actions: {q_values.flatten()}")
    print(f"Network architecture:\n{test_network.network}")