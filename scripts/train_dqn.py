"""Training script for DQN agent on customer offer environment"""

import argparse
import sys
import random
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add source directory to python path so we can import our modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent

def main():
    """Parse command line arguments and start training"""
    parser = argparse.ArgumentParser(description="Train DQN Agent for customer offer recommendations")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max_t", type=int, default=100, help="Max timesteps per episode (customer interactions)")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--eps_end", type=float, default=0.01, help="Minimum epsilon for exploration")
    parser.add_argument("--eps_decay", type=float, default=0.995, help="Multiplicative factor for epsilon decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Starting DQN training with {args.episodes} episodes...")

    # set up the environment
    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    state_size = env.observation_space.shape[0] # should be 5
    action_size = env.action_space.n    # should be 5

    print(f"Environment: state_size={state_size}, action_size={action_size}")

    # intialize DQN Agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        seed=args.seed
    )   
    scores =[]      # Store all episode scores for complete history
    scores_window = []      # Store last 100 scores for moving average
    eps = args.eps_start    # Epsilon for epsilon-greedy action selection

    print("Starting training loop...")

    # main training loop: each episode represents one customer's journey
    for episode in range(1, args.episodes + 1):
        # reset environment to start env customer interaction
        state, _ = env.reset(seed=args.seed + episode)
        score = 0   # Total revenue for this customer episode

        # Interact with customer for multiple time steps ( up to max_t)
        for t in range(args.max_t):
            # Select action using epsilon-greedy policy
            # High epsilon = more exploration, low epsilon = more exploitation
            action = agent.act(state, eps)

            # Take action in environment and get result
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # episode ended ?

            # Update agent with this experience (Learning happens here)
            agent.step(state, action, reward, next_state, done)

            # move to next state
            state = next_state
            score += reward    # Add revenue from this interaction to total

            if done:
                break   # episode ended early
        
        # Store episode results for tracking progress
        scores_window.append(score)
        scores.append(score)

        # Decay epsilon (reduce exploration over time)
        # Start with lots of exploration, end with mostly exploitation
        eps = max(args.eps_end, eps * args.eps_decay)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            avg_score = np.mean(scores_window[-100:])  # Average of last 100 episodes
            print(f'Episode {episode}\tAverage Score (last 100): {avg_score:.2f}\tEpsilon: {eps:.3f}')
    
    print('Training completed!')
    
    # Plot learning curve to visualize improvement over time
    plot_learning_curve(scores, args.episodes)
    
    # Save trained model for later use
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_customer_agent.pth')
    print('Model saved as dqn_customer_agent.pth')


def plot_learning_curve(scores, num_episodes):
    """Plot the learning curve starting from 0 for better interpretation."""
    # Calculate moving average
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = []
        episode_numbers = []
        
        for i in range(window_size, len(scores) + 1):
            moving_avg.append(np.mean(scores[i-window_size:i]))
            episode_numbers.append(i)  # Start from episode number, not 0
        
        plt.figure(figsize=(12, 6))
        plt.plot(episode_numbers, moving_avg, linewidth=2, color='blue')
        plt.title('DQN Training: Average Revenue per Customer (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Average Revenue per Customer ($)')
        plt.grid(True, alpha=0.3)
        
        # Start y-axis from 0 for better interpretation
        plt.ylim(bottom=0)
        
        # Add horizontal line showing initial performance
        if len(scores) > 0:
            initial_avg = np.mean(scores[:100])  # First 100 episodes
            plt.axhline(y=initial_avg, color='red', linestyle='--', 
                       label=f'Initial Average: ${initial_avg:.2f}', alpha=0.7)
        
        plt.legend()
        plt.savefig('dqn_training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Not enough episodes to plot moving average")


if __name__ == "__main__":
    main()