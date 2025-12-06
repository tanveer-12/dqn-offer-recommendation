"""Detailed DQN training with step-by-step visualization for each episode."""

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
    parser = argparse.ArgumentParser(description="Detailed DQN training with step-by-step episode analysis")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max_t", type=int, default=12, help="Max timesteps per episode (default horizon from config)")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--eps_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--loyalty_threshold", type=float, default=0.5, help="Loyalty threshold")
    parser.add_argument("--show_episodes", type=int, default=5, help="Number of episodes to show detailed output")
    args = parser.parse_args()

    print(f"Starting detailed DQN training with {args.episodes} episodes...")
    print(f"Loyalty threshold: {args.loyalty_threshold}")
    print(f"Showing detailed output for first {args.show_episodes} episodes")
    
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
        gamma=0.98,
        update_every=4
    )
    
    # Training loop with detailed tracking
    scores = []
    all_scores = []
    eps = args.eps_start
    
    print("Starting training loop...")
    print("Format: t=step action=action accepted=accepted revenue=revenue loyalty=loyalty recency=recency")
    
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        score = 0.0
        
        # Show detailed episode if within show_episodes range
        show_detailed = episode <= args.show_episodes
        
        if show_detailed:
            print(f"\nEpisode {episode}")
        
        for t in range(args.max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Show detailed step information for first few episodes
            if show_detailed:
                print(f"t={t:02d} action={action} accepted={info['accepted']} "
                      f"revenue={reward:6.2f} loyalty={info['loyalty_score']:.2f} "
                      f"recency={info['recency_of_purchase']}")
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        # Record episode results
        scores.append(score)
        all_scores.append(score)
        
        # Show total episode revenue for first few episodes
        if show_detailed:
            print(f"Total episode revenue: {score:.2f}")
        
        # Print summary every 10 episodes
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            print(f'Episode {episode}\tAverage Score (last 10): {avg_score:.2f}\tEpsilon: {eps:.3f}')
            scores = []  # Reset for next 10 episodes
        
        eps = max(args.eps_end, eps * args.eps_decay)
    
    print('Training completed!')
    
    # Create comprehensive visualization
    plot_detailed_learning_curve(all_scores, args.episodes, args.loyalty_threshold)
    
    # Save the detailed model
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_customer_agent_detailed.pth')
    print('Detailed model saved as dqn_customer_agent_detailed.pth')


def plot_detailed_learning_curve(scores, num_episodes, loyalty_threshold):
    """Plot learning curve starting from 0 with detailed information."""
    plt.figure(figsize=(15, 8))
    
    # Plot individual scores
    episode_numbers = list(range(1, len(scores) + 1))
    plt.plot(episode_numbers, scores, alpha=0.3, color='lightblue', label='Individual Episode Revenue', linewidth=0.5)
    
    # Add moving average line
    if len(scores) >= 20:
        moving_avg = []
        avg_episode_nums = []
        for i in range(20, len(scores) + 1):
            moving_avg.append(np.mean(scores[i-20:i]))
            avg_episode_nums.append(i)
        
        plt.plot(avg_episode_nums, moving_avg, color='blue', linewidth=2, label='20-episode Moving Average')
    
    plt.title(f'DQN Training: Revenue per Customer Episode\n(Loyalty Threshold: {loyalty_threshold})')
    plt.xlabel('Episode')
    plt.ylabel('Revenue per Customer ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Start from 0 on y-axis
    plt.ylim(bottom=0)
    
    # Add reference lines for business context
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Baseline Revenue ($50)')
    plt.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Good Performance ($100)')
    
    plt.savefig('detailed_dqn_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()