"""Experience replay buffer for DQN training."""

import random
import numpy as np
from collections import deque, namedtuple

# Define the structure of each experience tuple
# Why namedtuple: Clean way to store (state, action, reward, next_state, done) together
Experience = namedtuple('Experience', 
                       field_names=['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """
        Initialize the replay buffer.
        Why we need this:
        - Store past customer interactions for learning
        - Break correlation between consecutive experiences (sequential data is correlated)
        - Allow agent to learn from same experiences multiple times
        - Sample randomly for more diverse training batches

        Args:
            buffer_size: Maximum number of experiences to store (e.g., 100,000)
            batch_size: Number of experiences to sample for each training step (e.g., 32)
            seed: Random seed for reproducible sampling
        """
        # Create circular buffer that automatically removes oldest when full
        # Why circular: Efficient memory usage, always keeps most recent experiences
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience  # Store the namedtuple constructor
        self.seed = random.seed(seed)  # Set random seed for reproducible sampling
        print(f"ReplayBuffer initialized: size={buffer_size}, batch={batch_size}")

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        
        Why we store this:
        - state: Customer state before action (what the agent observed)
        - action: What offer the agent recommended
        - reward: Revenue earned from this interaction
        - next_state: Customer state after action (how customer changed)
        - done: Whether customer interaction ended
        
        Args:
            state: Current customer state before action
            action: Action taken (0-4 for different offers)
            reward: Revenue earned from this interaction
            next_state: State after action
            done: Whether episode ended
        """
        # Create experience tuple and add to memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)