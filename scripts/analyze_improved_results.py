"""Analyze improved DQN results to check for better offer distribution."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent

def load_improved_agent():
    """Load the improved trained agent."""
    env = CustomerOfferEnv(CustomerConfig(), seed=42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    
    try:
        agent.qnetwork_local.load_state_dict(torch.load('dqn_customer_agent_improved.pth'))
        print("Improved model loaded successfully!")
        return agent, env
    except FileNotFoundError:
        print("No improved model found. Please run improved training first.")
        return None, None

def test_improved_performance(agent, env, num_customers=200):
    """Test the improved agent's performance."""
    print(f"\nTesting improved agent performance on {num_customers} customers...")
    
    results = {
        'total_revenue': [],
        'acceptance_rates': [],
        'offer_distribution': {i: 0 for i in range(5)},
        'cost_effectiveness': []  # New metric
    }
    
    for customer_id in range(num_customers):
        state, _ = env.reset(seed=42 + customer_id)
        customer_revenue = 0
        customer_acceptances = 0
        customer_interactions = 0
        
        for t in range(50):
            action = agent.act(state, eps=0.0)
            results['offer_distribution'][action] += 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            customer_revenue += reward
            if info['accepted']:
                customer_acceptances += 1
            customer_interactions += 1
            
            state = next_state
            if done:
                break
        
        results['total_revenue'].append(customer_revenue)
        acceptance_rate = customer_acceptances / max(1, customer_interactions)
        results['acceptance_rates'].append(acceptance_rate)
    
    return results

def visualize_improved_results(results):
    """Create visualizations for improved results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Revenue distribution
    axes[0, 0].hist(results['total_revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(results['total_revenue']), color='red', linestyle='--',
                      label=f'Average: ${np.mean(results["total_revenue"]):.2f}')
    axes[0, 0].set_xlabel('Revenue per Customer')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].set_title('Revenue Distribution (Improved Agent)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Acceptance rate distribution
    axes[0, 1].hist(results['acceptance_rates'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(results['acceptance_rates']), color='red', linestyle='--',
                      label=f'Average: {np.mean(results["acceptance_rates"]):.3f}')
    axes[0, 1].set_xlabel('Offer Acceptance Rate')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Offer Acceptance Rate Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Improved offer distribution
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Calculate percentages
    total_offers = sum(results['offer_distribution'].values())
    percentages = [results['offer_distribution'][i] / total_offers * 100 for i in range(5)]
    
    bars = axes[1, 0].bar(offer_names, percentages, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Offer Type')
    axes[1, 0].set_ylabel('Percentage of Total Offers (%)')
    axes[1, 0].set_title('Offer Distribution (Should be more balanced)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Revenue vs Acceptance Rate
    axes[1, 1].scatter(results['acceptance_rates'], results['total_revenue'], 
                       alpha=0.6, color='coral')
    axes[1, 1].set_xlabel('Acceptance Rate')
    axes[1, 1].set_ylabel('Total Revenue')
    axes[1, 1].set_title('Revenue vs Acceptance Rate (Improved)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== IMPROVED PERFORMANCE SUMMARY ===")
    print(f"Average Revenue per Customer: ${np.mean(results['total_revenue']):.2f}")
    print(f"Average Acceptance Rate: {np.mean(results['acceptance_rates']):.3f}")
    print(f"Total Offers Made: {sum(results['offer_distribution'].values())}")
    print(f"Offer Distribution (counts): {results['offer_distribution']}")
    
    # Calculate percentages
    for offer, count in results['offer_distribution'].items():
        pct = (count / sum(results['offer_distribution'].values())) * 100
        offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
        print(f"{offer_names[offer]}: {pct:.1f}%")

def main():
    agent, env = load_improved_agent()
    if agent is None:
        return
    
    results = test_improved_performance(agent, env, num_customers=200)
    visualize_improved_results(results)

if __name__ == "__main__":
    main()