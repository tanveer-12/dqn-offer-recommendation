"""Analyze balanced DQN results."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent

def load_balanced_agent():
    """Load the balanced trained agent."""
    env = CustomerOfferEnv(CustomerConfig(), seed=42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    
    try:
        agent.qnetwork_local.load_state_dict(torch.load('dqn_customer_agent_balanced.pth'))
        print("Balanced model loaded successfully!")
        return agent, env
    except FileNotFoundError:
        print("No balanced model found. Please run balanced training first.")
        return None, None

def test_balanced_performance(agent, env, num_customers=200):
    """Test the balanced agent's performance."""
    print(f"\nTesting balanced agent performance on {num_customers} customers...")
    
    results = {
        'total_revenue': [],
        'acceptance_rates': [],
        'offer_distribution': {i: 0 for i in range(5)},
        'loyalty_changes': [],
        'customer_types': {'high_loyalty': [], 'low_loyalty': [], 'high_spend': [], 'low_spend': []}
    }
    
    for customer_id in range(num_customers):
        initial_state, _ = env.reset(seed=42 + customer_id)
        state = initial_state.copy()
        customer_revenue = 0
        customer_acceptances = 0
        customer_interactions = 0
        initial_loyalty = initial_state[0]
        initial_spend = initial_state[1]
        
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
        results['loyalty_changes'].append(state[0] - initial_loyalty)
        
        # Categorize customers
        if initial_loyalty > 0.6:
            results['customer_types']['high_loyalty'].append(customer_revenue)
        else:
            results['customer_types']['low_loyalty'].append(customer_revenue)
            
        if initial_spend > 80:
            results['customer_types']['high_spend'].append(customer_revenue)
        else:
            results['customer_types']['low_spend'].append(customer_revenue)
    
    return results

def visualize_balanced_results(results):
    """Create visualizations for balanced results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Revenue distribution
    axes[0, 0].hist(results['total_revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(results['total_revenue']), color='red', linestyle='--',
                      label=f'Average: ${np.mean(results["total_revenue"]):.2f}')
    axes[0, 0].set_xlabel('Revenue per Customer')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].set_title('Revenue Distribution (Balanced Agent)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Offer distribution
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    total_offers = sum(results['offer_distribution'].values())
    percentages = [results['offer_distribution'][i] / total_offers * 100 for i in range(5)]
    
    bars = axes[0, 1].bar(offer_names, percentages, color=colors, alpha=0.7)
    axes[0, 1].set_xlabel('Offer Type')
    axes[0, 1].set_ylabel('Percentage of Total Offers (%)')
    axes[0, 1].set_title('Offer Distribution (Should be balanced)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Acceptance rate distribution
    axes[0, 2].hist(results['acceptance_rates'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].axvline(np.mean(results['acceptance_rates']), color='red', linestyle='--',
                      label=f'Average: {np.mean(results["acceptance_rates"]):.3f}')
    axes[0, 2].set_xlabel('Offer Acceptance Rate')
    axes[0, 2].set_ylabel('Number of Customers')
    axes[0, 2].set_title('Offer Acceptance Rate Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Customer type performance
    high_loyalty_avg = np.mean(results['customer_types']['high_loyalty']) if results['customer_types']['high_loyalty'] else 0
    low_loyalty_avg = np.mean(results['customer_types']['low_loyalty']) if results['customer_types']['low_loyalty'] else 0
    high_spend_avg = np.mean(results['customer_types']['high_spend']) if results['customer_types']['high_spend'] else 0
    low_spend_avg = np.mean(results['customer_types']['low_spend']) if results['customer_types']['low_spend'] else 0
    
    customer_types = ['High Loyalty', 'Low Loyalty', 'High Spend', 'Low Spend']
    revenues = [high_loyalty_avg, low_loyalty_avg, high_spend_avg, low_spend_avg]
    
    bars = axes[1, 0].bar(customer_types, revenues, color=['green', 'red', 'blue', 'orange'], alpha=0.7)
    axes[1, 0].set_xlabel('Customer Type')
    axes[1, 0].set_ylabel('Average Revenue')
    axes[1, 0].set_title('Revenue by Customer Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Loyalty changes
    axes[1, 1].hist(results['loyalty_changes'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(np.mean(results['loyalty_changes']), color='red', linestyle='--',
                      label=f'Average: {np.mean(results["loyalty_changes"]):.3f}')
    axes[1, 1].set_xlabel('Loyalty Change')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Loyalty Change Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""
    === BALANCED PERFORMANCE SUMMARY ===
    Total Customers: {len(results['total_revenue'])}
    
    Revenue Metrics:
    - Average Revenue: ${np.mean(results['total_revenue']):.2f}
    - Best Customer: ${np.max(results['total_revenue']):.2f}
    - Worst Customer: ${np.min(results['total_revenue']):.2f}
    
    Customer Type Performance:
    - High Loyalty: ${high_loyalty_avg:.2f}
    - Low Loyalty: ${low_loyalty_avg:.2f}
    - High Spend: ${high_spend_avg:.2f}
    - Low Spend: ${low_spend_avg:.2f}
    
    Offer Distribution:
    - No Offer: {results['offer_distribution'][0]} ({results['offer_distribution'][0]/sum(results['offer_distribution'].values())*100:.1f}%)
    - 5% Disc: {results['offer_distribution'][1]} ({results['offer_distribution'][1]/sum(results['offer_distribution'].values())*100:.1f}%)
    - 10% Disc: {results['offer_distribution'][2]} ({results['offer_distribution'][2]/sum(results['offer_distribution'].values())*100:.1f}%)
    - BOGO: {results['offer_distribution'][3]} ({results['offer_distribution'][3]/sum(results['offer_distribution'].values())*100:.1f}%)
    - Premium: {results['offer_distribution'][4]} ({results['offer_distribution'][4]/sum(results['offer_distribution'].values())*100:.1f}%)
    """
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('balanced_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    agent, env = load_balanced_agent()
    if agent is None:
        return
    
    results = test_balanced_performance(agent, env, num_customers=200)
    visualize_balanced_results(results)

if __name__ == "__main__":
    main()