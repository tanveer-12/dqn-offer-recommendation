"""Cafe-based recommendation system training with loyalty threshold and detailed logging."""

import argparse
import sys
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For cluster compatibility
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent

def main():
    parser = argparse.ArgumentParser(description="Cafe-based personalized offer recommendation system")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--max_t", type=int, default=12, help="Max timesteps per episode (monthly visits)")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--eps_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--loyalty_threshold", type=float, default=0.5, help="Loyalty threshold")
    parser.add_argument("--show_episodes", type=int, default=50, help="Number of episodes to show detailed logs")
    args = parser.parse_args()

    print(f"Starting Cafe Recommendation System Training...")
    print(f"Total customers: {args.episodes}")
    print(f"Monthly visits per customer: {args.max_t}")
    print(f"Loyalty threshold: {args.loyalty_threshold}")
    print(f"Training for {args.episodes} episodes to learn optimal patterns")
    
    # Set up environment
    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize DQN agent with parameters for cafe environment
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        seed=args.seed,
        lr=args.lr,
        gamma=0.98,  # High gamma for long-term value optimization
        update_every=4,
        buffer_size=int(1e5),
        batch_size=64
    )
    # Initialize tracking variables
    all_scores = []
    training_curve = []
    print("\nStarting training loop...")
    print("Format: t=step action=action accepted=accepted revenue=revenue loyalty=loyalty recency=recency")
    print("=" * 100)
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        
        # Show detailed logs for first few episodes
        show_detailed = episode <= args.show_episodes
        
        if show_detailed:
            print(f"\nEpisode {episode} (Customer #{episode})")
        
        score = 0.0
        
        for t in range(args.max_t):
            action = agent.act(state, args.eps_start * (args.eps_decay ** episode))  # Dynamic epsilon
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Log each step if showing detailed
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
        all_scores.append(score)
        
        # Show total episode revenue for detailed episodes
        if show_detailed:
            customer_type = "LOYAL" if info['loyalty_score'] > args.loyalty_threshold else "NOT LOYAL"
            print(f"Total episode revenue: {score:.2f} | Final loyalty: {info['loyalty_score']:.2f} | Type: {customer_type}")
        
        # Track training progress for curve
        if episode % 10 == 0:  # Track every 10 episodes
            avg_score = np.mean(all_scores[-10:]) if len(all_scores) >= 10 else np.mean(all_scores)
            training_curve.append((episode, avg_score))
        
        # Print progress every 500 episodes
        if episode % 200 == 0:
            avg_score = np.mean(all_scores[-200:]) if len(all_scores) >= 200 else np.mean(all_scores)
            print(f'Episode {episode}\tAverage Score (last 200): {avg_score:.2f}')
    
    print('\n' + '=' * 100)
    print('TRAINING COMPLETED!')
    # Create training curve plot
    create_training_curve(training_curve, args)
    
    # Save the trained model
    torch.save(agent.qnetwork_local.state_dict(), 'cafe_recommendation_model.pth')
    print('Model saved as cafe_recommendation_model.pth')
    
    # Print final statistics
    print_final_statistics(all_scores, args)

def create_training_curve(training_curve, args):
    """Create and save training curve starting from 0."""
    if len(training_curve) == 0:
        return
    
    episodes, scores = zip(*training_curve)
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, scores, linewidth=2, color='blue', label='Average Score (10-episode window)')
    plt.title(f'Cafe Recommendation System Training Curve\n({args.episodes} episodes, Loyalty Threshold: {args.loyalty_threshold})')
    plt.xlabel('Episode')
    plt.ylabel('Average Revenue per Customer ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)  # Start from 0
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curve saved as training_curve.png")

def print_final_statistics(all_scores, args):
    """Print final training statistics."""
    print(f"\n{'='*50}")
    print("FINAL TRAINING STATISTICS")
    print(f"{'='*50}")
    
    total_revenue = sum(all_scores)
    avg_revenue = np.mean(all_scores)
    best_revenue = np.max(all_scores)
    worst_revenue = np.min(all_scores)
    
    print(f"Total customers processed: {len(all_scores)}")
    print(f"Total revenue generated: ${total_revenue:.2f}")
    print(f"Average revenue per customer: ${avg_revenue:.2f}")
    print(f"Best customer revenue: ${best_revenue:.2f}")
    print(f"Worst customer revenue: ${worst_revenue:.2f}")
    print(f"Revenue range: ${worst_revenue:.2f} - ${best_revenue:.2f}")
    
    # For cafe context (realistic spending)
    avg_daily_spending = avg_revenue / args.max_t  # Revenue per visit
    print(f"Average daily spending per customer: ${avg_daily_spending:.2f}")
    print(f"Average monthly spending per customer: ${avg_revenue:.2f}")


if __name__ == "__main__":
    main()