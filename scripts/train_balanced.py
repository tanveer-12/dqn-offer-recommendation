"""Balanced DQN training with proper exploration and reward shaping."""

import argparse
import sys
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser(description="Train balanced DQN agent for customer offer recommendations")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--max_t", type=int, default=100, help="Max timesteps per episode")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Minimum epsilon")  # Higher end epsilon
    parser.add_argument("--eps_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    print(f"Starting balanced DQN training with {args.episodes} episodes...")
    
    # Set up environment
    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: state_size={state_size}, action_size={action_size}")
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        seed=args.seed,
        lr=args.lr,
        gamma=0.98,  # Higher gamma for more long-term focus
        update_every=4  # Standard update frequency
    )
    
    # Training loop with better tracking
    scores = []
    scores_window = []
    eps = args.eps_start
    offer_usage = {i: 0 for i in range(5)}  # Track offer usage
    
    print("Starting training loop...")
    
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        score = 0
        
        for t in range(args.max_t):
            action = agent.act(state, eps)
            offer_usage[action] += 1  # Track this action
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(args.eps_end, eps * args.eps_decay)
        
        if episode % 500 == 0:
            avg_score = np.mean(scores_window[-100:])
            print(f'Episode {episode}\tAverage Score (last 100): {avg_score:.2f}\tEpsilon: {eps:.3f}')
            print(f'  Offer distribution: {offer_usage}')
            # Reset counter for next 500 episodes
            offer_usage = {i: 0 for i in range(5)}
    
    print('Training completed!')
    
    plot_learning_curve(scores, args.episodes)
    
    # Save the balanced model
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_customer_agent_balanced.pth')
    print('Balanced model saved as dqn_customer_agent_balanced.pth')


def plot_learning_curve(scores, num_episodes):
    """Plot the learning curve."""
    window_size = 100
    if len(scores) >= window_size:
        moving_avg = []
        episode_numbers = []
        
        for i in range(window_size, len(scores) + 1):
            moving_avg.append(np.mean(scores[i-window_size:i]))
            episode_numbers.append(i)
        
        plt.figure(figsize=(12, 6))
        plt.plot(episode_numbers, moving_avg, linewidth=2, color='blue')
        plt.title('Balanced DQN Training: Average Revenue per Customer (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Average Revenue per Customer ($)')
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        plt.savefig('balanced_dqn_training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Not enough episodes to plot moving average")


if __name__ == "__main__":
    main()