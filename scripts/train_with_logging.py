"""DQN training with comprehensive logging and customer analysis."""

import argparse
import sys
import random
from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent

def main():
    parser = argparse.ArgumentParser(description="DQN training with comprehensive logging")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--max_t", type=int, default=12, help="Max timesteps per episode")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--eps_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--loyalty_threshold", type=float, default=0.5, help="Loyalty threshold")
    parser.add_argument("--show_episodes", type=int, default=10, help="Number of episodes to show detailed output")
    args = parser.parse_args()

    print(f"Starting DQN training with comprehensive logging...")
    print(f"Total episodes: {args.episodes}")
    print(f"Max interactions per customer: {args.max_t}")
    print(f"Loyalty threshold: {args.loyalty_threshold}")
    print(f"Tracking {args.episodes} customers over {args.max_t} interactions each")

    # Set up environment
    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        seed=args.seed,
        lr=args.lr,
        gamma=0.98,
        update_every=4
    )
    
    # Initialize tracking variables
    all_episode_data = []
    all_scores = []
    eps = args.eps_start
    
    # Initialize offer acceptance tracking
    offer_acceptance_counts = {i: {'accepted': 0, 'rejected': 0} for i in range(5)}
    loyalty_tracking = {'initial': [], 'final': [], 'changes': []}
    
    print("Starting training loop...")
    print("=" * 80)
    
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        initial_loyalty = state[0]
        initial_spend = state[1]
        
        episode_data = {
            'episode': episode,
            'initial_loyalty': initial_loyalty,
            'initial_spend': initial_spend,
            'steps': [],
            'total_revenue': 0.0,
            'final_loyalty': 0.0,
            'loyalty_change': 0.0,
            'customer_type': 'not_loyal',
            'offer_sequence': []
        }
        
        score = 0.0
        
        # Show detailed episode if within show_episodes range
        show_detailed = episode <= args.show_episodes
        
        if show_detailed:
            print(f"\nEpisode {episode} (Customer #{episode})")
            print(f"Initial: Loyalty={initial_loyalty:.2f}, Spend=${initial_spend:.2f}")
        
        for t in range(args.max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track offer acceptance
            if info['accepted']:
                offer_acceptance_counts[action]['accepted'] += 1
            else:
                offer_acceptance_counts[action]['rejected'] += 1
            
            # Store step information
            step_info = {
                't': t,
                'action': action,
                'accepted': info['accepted'],
                'revenue': reward,
                'loyalty': info['loyalty_score'],
                'recency': info['recency_of_purchase']
            }
            episode_data['steps'].append(step_info)
            episode_data['offer_sequence'].append(action)
            
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
        
        # Finalize episode data
        final_loyalty = episode_data['steps'][-1]['loyalty'] if episode_data['steps'] else initial_loyalty
        loyalty_change = final_loyalty - initial_loyalty
        
        episode_data['total_revenue'] = score
        episode_data['final_loyalty'] = final_loyalty
        episode_data['loyalty_change'] = loyalty_change
        episode_data['customer_type'] = 'loyal' if final_loyalty > args.loyalty_threshold else 'not_loyal'
        
        # Store episode data
        all_episode_data.append(episode_data)
        all_scores.append(score)
        
        # Update loyalty tracking
        loyalty_tracking['initial'].append(initial_loyalty)
        loyalty_tracking['final'].append(final_loyalty)
        loyalty_tracking['changes'].append(loyalty_change)
        
        # Show total episode revenue for first few episodes
        if show_detailed:
            print(f"Total episode revenue: {score:.2f}")
            print(f"Final loyalty: {final_loyalty:.2f} (change: {loyalty_change:+.2f})")
            print(f"Customer type: {episode_data['customer_type'].upper()}")
            print(f"Offer sequence: {episode_data['offer_sequence']}")
        
        # Print summary every 50 episodes
        if episode % 50 == 0:
            avg_score = np.mean(all_scores[-50:]) if len(all_scores) >= 50 else np.mean(all_scores)
            print(f'\n--- Episode {episode} Summary ---')
            print(f'Average Score (last 50): {avg_score:.2f} | Epsilon: {eps:.3f}')
            print(f'Customers processed: {episode}/{args.episodes}')
            print(f'Progress: {episode/args.episodes*100:.1f}%')
        
        eps = max(args.eps_end, eps * args.eps_decay)
    
    print('\n' + '=' * 80)
    print('TRAINING COMPLETED!')
    
    # Create comprehensive analysis
    create_comprehensive_analysis(all_episode_data, offer_acceptance_counts, loyalty_tracking, args)
    
    # Save the model
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_customer_agent_logged.pth')
    print('Model saved as dqn_customer_agent_logged.pth')

def create_comprehensive_analysis(episode_data, offer_acceptance_counts, loyalty_tracking, args):
    """Create comprehensive analysis and visualizations."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 80)
    
    # Basic statistics
    total_customers = len(episode_data)
    total_revenue = sum([ep['total_revenue'] for ep in episode_data])
    avg_revenue = np.mean([ep['total_revenue'] for ep in episode_data])
    avg_loyalty_change = np.mean(loyalty_tracking['changes'])
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total customers analyzed: {total_customers}")
    print(f"Total revenue generated: ${total_revenue:.2f}")
    print(f"Average revenue per customer: ${avg_revenue:.2f}")
    print(f"Average loyalty change per customer: {avg_loyalty_change:+.3f}")
    
    # Customer classification
    loyal_customers = [ep for ep in episode_data if ep['customer_type'] == 'loyal']
    not_loyal_customers = [ep for ep in episode_data if ep['customer_type'] == 'not_loyal']
    
    print(f"\n👥 CUSTOMER CLASSIFICATION:")
    print(f"Loyal customers (loyalty > {args.loyalty_threshold}): {len(loyal_customers)} ({len(loyal_customers)/total_customers*100:.1f}%)")
    print(f"Not loyal customers: {len(not_loyal_customers)} ({len(not_loyal_customers)/total_customers*100:.1f}%)")
    
    # Revenue by customer type
    loyal_revenue = np.mean([ep['total_revenue'] for ep in loyal_customers]) if loyal_customers else 0
    not_loyal_revenue = np.mean([ep['total_revenue'] for ep in not_loyal_customers]) if not_loyal_customers else 0
    
    print(f"\n💰 REVENUE BY CUSTOMER TYPE:")
    print(f"Average revenue - Loyal customers: ${loyal_revenue:.2f}")
    print(f"Average revenue - Not loyal customers: ${not_loyal_revenue:.2f}")
    
    # Offer acceptance analysis
    print(f"\n🎯 OFFER ACCEPTANCE ANALYSIS:")
    offer_names = ['No Offer', '5% Discount', '10% Discount', 'BOGO', 'Premium']
    for i, name in enumerate(offer_names):
        accepted = offer_acceptance_counts[i]['accepted']
        rejected = offer_acceptance_counts[i]['rejected']
        total = accepted + rejected
        acceptance_rate = accepted / total * 100 if total > 0 else 0
        print(f"{name}: {accepted} accepted, {rejected} rejected, {acceptance_rate:.1f}% acceptance rate")
    
    # Loyalty analysis
    print(f"\n🏆 LOYALTY ANALYSIS:")
    print(f"Average initial loyalty: {np.mean(loyalty_tracking['initial']):.3f}")
    print(f"Average final loyalty: {np.mean(loyalty_tracking['final']):.3f}")
    print(f"Average loyalty improvement: {avg_loyalty_change:+.3f}")
    print(f"Customers with loyalty improvement: {sum(1 for change in loyalty_tracking['changes'] if change > 0)}/{total_customers}")
    print(f"Customers with loyalty decline: {sum(1 for change in loyalty_tracking['changes'] if change < 0)}/{total_customers}")
    
    # Create visualizations
    create_detailed_visualizations(episode_data, offer_acceptance_counts, loyalty_tracking, args)
    
    # Save detailed data
    save_detailed_data(episode_data, offer_acceptance_counts, loyalty_tracking, args)

def create_detailed_visualizations(episode_data, offer_acceptance_counts, loyalty_tracking, args):
    """Create detailed visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Revenue distribution
    revenues = [ep['total_revenue'] for ep in episode_data]
    axes[0, 0].hist(revenues, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(revenues), color='red', linestyle='--', 
                      label=f'Average: ${np.mean(revenues):.2f}')
    axes[0, 0].set_xlabel('Revenue per Customer ($)')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].set_title('Revenue Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Loyalty changes
    axes[0, 1].hist(loyalty_tracking['changes'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(loyalty_tracking['changes']), color='red', linestyle='--',
                      label=f'Average: {np.mean(loyalty_tracking["changes"]):+.3f}')
    axes[0, 1].set_xlabel('Loyalty Change')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Loyalty Change Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Offer acceptance rates
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    acceptance_rates = []
    for i in range(5):
        total = offer_acceptance_counts[i]['accepted'] + offer_acceptance_counts[i]['rejected']
        rate = offer_acceptance_counts[i]['accepted'] / total if total > 0 else 0
        acceptance_rates.append(rate * 100)
    
    bars = axes[0, 2].bar(offer_names, acceptance_rates, 
                         color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    axes[0, 2].set_xlabel('Offer Type')
    axes[0, 2].set_ylabel('Acceptance Rate (%)')
    axes[0, 2].set_title('Offer Acceptance Rates')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, rate in zip(bars, acceptance_rates):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Loyalty trajectory over time (first 20 customers)
    for i in range(min(20, len(episode_data))):
        steps = episode_data[i]['steps']
        if steps:
            loyalty_trajectory = [step['loyalty'] for step in steps]
            axes[1, 0].plot(range(len(loyalty_trajectory)), loyalty_trajectory, 
                           alpha=0.6, linewidth=1)
    axes[1, 0].axhline(y=args.loyalty_threshold, color='red', linestyle='--', alpha=0.8, 
                      label=f'Threshold ({args.loyalty_threshold})')
    axes[1, 0].set_xlabel('Interaction Step')
    axes[1, 0].set_ylabel('Loyalty Score')
    axes[1, 0].set_title('Loyalty Trajectories (First 20 Customers)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Revenue vs Initial Loyalty
    initial_loyalties = [ep['initial_loyalty'] for ep in episode_data]
    revenues = [ep['total_revenue'] for ep in episode_data]
    axes[1, 1].scatter(initial_loyalties, revenues, alpha=0.6, color='coral')
    axes[1, 1].set_xlabel('Initial Loyalty')
    axes[1, 1].set_ylabel('Total Revenue ($)')
    axes[1, 1].set_title('Revenue vs Initial Loyalty')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Customer type comparison
    loyal_revenues = [ep['total_revenue'] for ep in episode_data if ep['customer_type'] == 'loyal']
    not_loyal_revenues = [ep['total_revenue'] for ep in episode_data if ep['customer_type'] == 'not_loyal']
    axes[1, 2].boxplot([loyal_revenues, not_loyal_revenues], 
                      labels=['Loyal', 'Not Loyal'])
    axes[1, 2].set_ylabel('Revenue per Customer ($)')
    axes[1, 2].set_title('Revenue: Loyal vs Not Loyal Customers')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_detailed_data(episode_data, offer_acceptance_counts, loyalty_tracking, args):
    """Save detailed data to files."""
    
    # Create summary statistics
    summary = {
        'parameters': {
            'total_episodes': len(episode_data),
            'max_interactions_per_customer': args.max_t,
            'loyalty_threshold': args.loyalty_threshold,
            'total_customers': len(episode_data),
            'total_revenue': sum([ep['total_revenue'] for ep in episode_data]),
            'avg_revenue_per_customer': np.mean([ep['total_revenue'] for ep in episode_data]),
            'avg_loyalty_change': np.mean(loyalty_tracking['changes'])
        },
        'customer_classification': {
            'loyal_customers_count': len([ep for ep in episode_data if ep['customer_type'] == 'loyal']),
            'not_loyal_customers_count': len([ep for ep in episode_data if ep['customer_type'] == 'not_loyal']),
            'loyal_customers_percentage': len([ep for ep in episode_data if ep['customer_type'] == 'loyal']) / len(episode_data) * 100
        },
        'offer_analysis': {
            'offer_acceptance_counts': offer_acceptance_counts,
            'offer_acceptance_rates': {}
        },
        'loyalty_analysis': {
            'avg_initial_loyalty': np.mean(loyalty_tracking['initial']),
            'avg_final_loyalty': np.mean(loyalty_tracking['final']),
            'avg_loyalty_change': np.mean(loyalty_tracking['changes']),
            'loyalty_improvement_count': sum(1 for change in loyalty_tracking['changes'] if change > 0),
            'loyalty_decline_count': sum(1 for change in loyalty_tracking['changes'] if change < 0)
        }
    }
    
    # Calculate acceptance rates
    for i in range(5):
        total = offer_acceptance_counts[i]['accepted'] + offer_acceptance_counts[i]['rejected']
        rate = offer_acceptance_counts[i]['accepted'] / total if total > 0 else 0
        summary['offer_analysis']['offer_acceptance_rates'][i] = rate
    
    # Save summary
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed episode data (first 100 episodes to avoid file being too large)
    with open('detailed_episode_data.json', 'w') as f:
        json.dump(episode_data[:100], f, indent=2)  # Save first 100 episodes
    
    print(f"\n💾 DATA SAVED:")
    print(f"- Training summary: training_summary.json")
    print(f"- Detailed episode data (first 100): detailed_episode_data.json")
    print(f"- Visualizations: comprehensive_analysis_visualizations.png")


if __name__ == "__main__":
    main()