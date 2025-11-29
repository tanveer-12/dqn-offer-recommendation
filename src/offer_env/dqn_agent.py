"""DQN Agent implementation for customer offer recommendation"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

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
        Initialize the DQN agent.
        
        Why two networks:
        - Local network: Makes decisions and gets updated frequently
        - Target network: Provides stable targets, updated slowly to prevent oscillations
        
        Args:
            state_size: Number of state features (5)
            action_size: Number of possible actions (5 offers)
            seed: Random seed for reproducibility
            buffer_size: Size of the replay buffer
            batch_size: Size of training batches
            gamma: Discount factor for future rewards (0.99 = very future-focused)
            lr: Learning rate for neural network optimizer
            update_every: How often to update the network (every 4 steps)
            tau: Soft update parameter for target network (1e-3 = slow updates)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.batch_size = batch_size
        self.gamma = gamma      # How much to value future rewards
        self.update_every = update_every
        self.tau = tau  # Rate of target network update
        
        print(f"DQNAgent initialized: state_size={state_size}, action_size={action_size}")
        print(f"Hyperparameters: gamma={gamma}, lr={lr}, update_every={update_every}")

        # Create two identical neural networks
        # Local network: Makes decisions and gets updated frequently
        self.qnetwork_local = QNetwork(state_size, action_size)
        # Target network: Provides stable targets for learning
        self.qnetwork_target = QNetwork(state_size, action_size)
        # Adma Optimizer: Efficient and works well for neural networks
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Experience replay buffer for storing customer interactions
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

        # Initialize time step counter for update scheduling
        self.t_step = 0
    

    def step(self, state, action, reward, next_state, done):
        """
        Add experience to memory and learn from batch.

        This is called after each customer interaction
        Why update_every: Update network every N steps for stability
        """
        # Save this customer interation in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every steps(e.g., every 4 customer interactions)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Learn if enough samples in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.0):
        """
        Return action for given state using epsilon-greedy policy
        Why epsilon-greedy:
        - Exploration (random actions): Discover new strategies
        - Exploitation (best Q-values): Use known good strategies
        - Start with high epsilon, decrease over time

        Args:
            state: current customer state
            eps: epsilon for epsilon-greedy exploration (0.0 = no exploration)

        Returns:
            Action (0-4) to recommend to customer
        """
        # Convert numpy state to Pytorch tensor and add batch dimension
        state = torch.from_numpy(state).float().unsqueeze(0)     # shape : [1, 5]

        # set network to evaluation mode (no dropout, batch norm uses stored stats)
        self.qnetwork_local.eval()
        with torch.no_grad():       # Don't compute gradients for inference
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()     # Set back to training mode

        # epsilon-greedy action selection
        if random.random() > eps:
            # Choose action with highest Q-value (exploration)
            # argmax finds index of highest value
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Choose random action ( exploration )
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences):
        """
        Updates value parameters using batch of experiences
        This implements the Bellman Equation: Q(s,a) = r + gamma * max(Q(s', a'))
        Where s' is next state and a' are possible next actions
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target network
        # detach() prevents gradients form flowing back to target network
        # max(1)[0] gets maximum value along action dimension (dim=1)
        # unsqueeze(1) adds dimension to match rewards shape [batch_size, 1]
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Comopute Q targets for current states: r + gammma * max(Q(s', a')) if not done
        # if done=True, future reward is 0 ( episode ended)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model (current predictions)
        # gather(1, actions) selects Q-value for the action that was actually taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute Loss (how wrong our predictions were)
        # MSE Loss: (predicted - target)^2
        loss = F.mse_loss(Q_expected, Q_targets)

        # minimize loss (update neural network weights)
        self.optimizer.zero_grad()  #Clear previous gradients
        loss.backward()     # compute gradients
        self.optimizer.step()   # update weights

        # Update target network periodically using soft update
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters: 0_target = tau*0_local + (1-tau)*0_target

        Why soft update: gradually move target n/w toward local nw
        Prevents target values from changing too rapidly, which causes instability
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # Update each parameter slowlu toward local network values
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

# test the agent
if __name__ == "__main__":
    import random

    # Create agent
    agent = DQNAgent(state_size=5, action_size=5, seed=42)

    # Test action selection (exploration vs exploitation)
    test_state = np.random.random(5)

    # with high epsilon ( exploration )
    action_explore = agent.act(test_state, eps=1.0)
    print(f"Exploration action: {action_explore}")

    # With low epsilon (exploitation)
    action_exploit = agent.act(test_state, eps=0.0)
    print(f"Exploitation action: {action_exploit}")

    # test learning step
    fake_experience = (
        torch.randn(32, 5),  # 32 states with 5 features each
        torch.randint(0, 5, (32, 1)),   # 32 random actions
        torch.randn(32, 1),     # 32 rewards
        torch.randn(32, 5),     # 32 next steps
        torch.zeros(32, 1)    # 32 done flags (all False)
    )

    agent.learn(fake_experience)
    print(f"Learning step completed successfully")