"""Analyze and visualize DQN training results"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent

def load_trained_agent():
    """Load the trained agent for analysis"""
    env = CustomerOfferEnv(CustomerConfig(), seed=42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    try:
        agent.qnetwork_local.load_state_dict(torch.load('dqn_customer_agent.pth'))
        print("Trained model loaded successfully!")
        return agent, env
    except FileNotFoundError:
        print("No trained model found. Please run training first.")
        return None, None


def test_agent_performance(agent, env, num_customers=100):
    """Test the trained agent's performance across multiple customers."""
    print(f"\nTesting agent performance on {num_customers} customers...")
    
    total_revenue = []
    acceptance_rates = []
    offer_distribution = {i: 0 for i in range(5)}
    customer_states = []
    
    for customer_id in range(num_customers):
        state, _ = env.reset(seed=42 + customer_id)
        customer_revenue = 0
        customer_acceptances = 0
        customer_interactions = 0
        
        for t in range(50):  # 50 interactions per customer
            action = agent.act(state, eps=0.0)  # No exploration, pure exploitation
            offer_distribution[action] += 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            customer_revenue += reward
            if info['accepted']:
                customer_acceptances += 1
            customer_interactions += 1
            
            state = next_state
            if done:
                break
        
        total_revenue.append(customer_revenue)
        acceptance_rate = customer_acceptances / max(1, customer_interactions)
        acceptance_rates.append(acceptance_rate)
    
    return {
        'total_revenue': total_revenue,
        'acceptance_rates': acceptance_rates,
        'offer_distribution': offer_distribution,
        'avg_revenue': np.mean(total_revenue),
        'avg_acceptance_rate': np.mean(acceptance_rates)
    }


def visualize_results(results):
    """Create comprehensive visualizations of agent performance."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Revenue distribution
    axes[0, 0].hist(results['total_revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(results['avg_revenue'], color='red', linestyle='--', 
                      label=f'Average: ${results["avg_revenue"]:.2f}')
    axes[0, 0].set_xlabel('Revenue per Customer')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].set_title('Distribution of Revenue per Customer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Acceptance rate distribution
    axes[0, 1].hist(results['acceptance_rates'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(results['avg_acceptance_rate'], color='red', linestyle='--',
                      label=f'Average: {results["avg_acceptance_rate"]:.3f}')
    axes[0, 1].set_xlabel('Offer Acceptance Rate')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Distribution of Offer Acceptance Rates')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Offer distribution
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    axes[1, 0].bar(offer_names, list(results['offer_distribution'].values()), 
                   color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    axes[1, 0].set_xlabel('Offer Type')
    axes[1, 0].set_ylabel('Number of Times Used')
    axes[1, 0].set_title('Distribution of Offers Used by Agent')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Revenue vs Acceptance Rate scatter
    axes[1, 1].scatter(results['acceptance_rates'], results['total_revenue'], 
                       alpha=0.6, color='coral')
    axes[1, 1].set_xlabel('Acceptance Rate')
    axes[1, 1].set_ylabel('Total Revenue')
    axes[1, 1].set_title('Revenue vs Acceptance Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"Average Revenue per Customer: ${results['avg_revenue']:.2f}")
    print(f"Average Acceptance Rate: {results['avg_acceptance_rate']:.3f}")
    print(f"Total Offers Made: {sum(results['offer_distribution'].values())}")
    print(f"Offer Distribution: {results['offer_distribution']}")

def analyze_customer_segments(agent, env):
    """Analyze how the agent treats different types of customers."""
    print("\n=== CUSTOMER SEGMENT ANALYSIS ===")
    
    # Test different customer types
    customer_types = {
        'High Loyalty': [0.8, 100, -1, 0, 1],  # High loyalty, high spender
        'Low Loyalty': [0.2, 50, -1, 0, 3],    # Low loyalty, moderate spender  
        'Inactive': [0.5, 75, -1, 0, 10],      # Moderate loyalty, long since purchase
        'High Spender': [0.6, 150, -1, 0, 2], # High spender, moderate loyalty
    }
    
    for customer_type, base_state in customer_types.items():
        print(f"\n{customer_type} Customer Analysis:")
        
        # Simulate multiple interactions for this customer type
        total_revenue = 0
        offers_used = {i: 0 for i in range(5)}
        
        for trial in range(20):  # Run 20 trials to get average behavior
            state = np.array(base_state)
            env_state, _ = env.reset(seed=1000 + trial)
            # Set initial state to our test state (this requires modifying env if needed)
            
            for step in range(10):  # 10 interactions per customer
                action = agent.act(state, eps=0.0)
                offers_used[action] += 1
                
                # Simulate the environment step (simplified)
                # In practice, you'd need to properly reset to your test state
                break  # For now, just test initial action choice
        
        print(f"  Offers typically chosen: {offers_used}")

def main():
    agent, env = load_trained_agent()
    if agent is None:
        return
    
    # Test performance
    results = test_agent_performance(agent, env, num_customers=200)
    
    # Create visualizations
    visualize_results(results)
    
    # Analyze customer segments
    analyze_customer_segments(agent, env)

if __name__ == "__main__":
    main()