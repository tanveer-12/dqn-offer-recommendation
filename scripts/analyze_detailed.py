"""Detailed analysis of DQN agent with loyalty threshold and business metrics."""

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
    """Load the trained agent for analysis."""
    env = CustomerOfferEnv(CustomerConfig(), seed=42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    
    # Try different model file names
    model_files = [
        'dqn_customer_agent_detailed.pth',
        'dqn_customer_agent_final.pth',
        'dqn_customer_agent_balanced.pth',
        'dqn_customer_agent_improved.pth',
        'dqn_customer_agent.pth'
    ]
    
    for file in model_files:
        try:
            agent.qnetwork_local.load_state_dict(torch.load(file))
            print(f"Trained model '{file}' loaded successfully!")
            return agent, env
        except FileNotFoundError:
            continue
    
    print("No trained model found. Please run training first.")
    return None, None

def analyze_customer_journeys(agent, env, num_customers=100, loyalty_threshold=0.5):
    """Analyze detailed customer journeys with loyalty classification."""
    print(f"\n=== DETAILED CUSTOMER JOURNEY ANALYSIS ===")
    print(f"Loyalty Threshold: {loyalty_threshold}")
    
    results = {
        'loyal_customers': {'revenue': [], 'loyalty_changes': [], 'offer_dist': {i: 0 for i in range(5)}},
        'non_loyal_customers': {'revenue': [], 'loyalty_changes': [], 'offer_dist': {i: 0 for i in range(5)}},
        'offer_distribution': {i: 0 for i in range(5)},
        'loyalty_trajectories': [],
        'total_revenue': 0
    }
    
    for customer_id in range(num_customers):
        state, _ = env.reset(seed=42 + customer_id)
        initial_loyalty = state[0]
        customer_revenue = 0.0
        customer_loyalty_trajectory = [initial_loyalty]
        
        # Simulate customer journey with detailed step tracking
        for t in range(15):  # 15 interactions per customer
            action = agent.act(state, eps=0.0)  # No exploration
            results['offer_distribution'][action] += 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            customer_revenue += reward
            customer_loyalty_trajectory.append(info['loyalty_score'])
            
            state = next_state
            if done:
                break
        
        # Classify customer based on final loyalty
        final_loyalty = customer_loyalty_trajectory[-1]
        if final_loyalty > loyalty_threshold:
            results['loyal_customers']['revenue'].append(customer_revenue)
            results['loyal_customers']['loyalty_changes'].append(final_loyalty - initial_loyalty)
            # Add to loyal customer offer distribution
            for offer, count in results['offer_distribution'].items():
                results['loyal_customers']['offer_dist'][offer] = count  # This will be overwritten, so we'll fix it later
        else:
            results['non_loyal_customers']['revenue'].append(customer_revenue)
            results['non_loyal_customers']['loyalty_changes'].append(final_loyalty - initial_loyalty)
        
        results['loyalty_trajectories'].append(customer_loyalty_trajectory)
        results['total_revenue'] += customer_revenue
    
    return results

def analyze_single_customer_detailed(agent, env, loyalty_threshold=0.5):
    """Analyze a single customer journey in detail."""
    print(f"\n=== DETAILED SINGLE CUSTOMER ANALYSIS ===")
    print(f"Loyalty Threshold: {loyalty_threshold}")
    
    state, _ = env.reset(seed=100)
    initial_loyalty = state[0]
    initial_spend = state[1]
    
    print(f"Initial State: Loyalty={initial_loyalty:.2f}, Spend=${initial_spend:.2f}")
    print(f"Customer Classification: {'LOYAL' if initial_loyalty > loyalty_threshold else 'NOT LOYAL'}")
    print("\nStep-by-step journey:")
    print("Step | Action | Accepted | Revenue | Loyalty | Recency | Category")
    print("-" * 70)
    
    total_revenue = 0.0
    step_count = 0
    
    for t in range(10):  # Show first 10 steps
        action = agent.act(state, eps=0.0)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_revenue += reward
        customer_category = "LOYAL" if info['loyalty_score'] > loyalty_threshold else "NOT LOYAL"
        
        print(f"{t:4d} | {action:6d} | {str(info['accepted']):8s} | {reward:7.2f} | {info['loyalty_score']:7.2f} | {info['recency_of_purchase']:7d} | {customer_category}")
        
        state = next_state
        step_count += 1
        if done:
            break
    
    final_loyalty = info['loyalty_score']
    loyalty_change = final_loyalty - initial_loyalty
    final_category = "LOYAL" if final_loyalty > loyalty_threshold else "NOT LOYAL"
    
    print("-" * 70)
    print(f"Final State: Loyalty={final_loyalty:.2f} ({'↑' if loyalty_change > 0 else '↓' if loyalty_change < 0 else '→'} {abs(loyalty_change):.2f})")
    print(f"Total Revenue: ${total_revenue:.2f}")
    print(f"Final Category: {final_category}")
    print(f"Revenue per Interaction: ${total_revenue/step_count:.2f}")

def visualize_detailed_results(results, loyalty_threshold):
    """Create detailed visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Revenue comparison: loyal vs non-loyal
    loyal_revenue = results['loyal_customers']['revenue']
    non_loyal_revenue = results['non_loyal_customers']['revenue']
    
    axes[0, 0].boxplot([loyal_revenue, non_loyal_revenue], labels=['Loyal', 'Non-Loyal'])
    axes[0, 0].set_ylabel('Revenue per Customer ($)')
    axes[0, 0].set_title('Revenue: Loyal vs Non-Loyal Customers')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Offer distribution
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    offer_counts = [results['offer_distribution'][i] for i in range(5)]
    axes[0, 1].bar(offer_names, offer_counts, 
                   color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    axes[0, 1].set_xlabel('Offer Type')
    axes[0, 1].set_ylabel('Number of Times Used')
    axes[0, 1].set_title('Overall Offer Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Loyalty changes
    loyal_changes = results['loyal_customers']['loyalty_changes']
    non_loyal_changes = results['non_loyal_customers']['loyalty_changes']
    axes[0, 2].hist([loyal_changes, non_loyal_changes], 
                   label=['Loyal Customers', 'Non-Loyal Customers'], 
                   alpha=0.7, bins=20)
    axes[0, 2].set_xlabel('Loyalty Change')
    axes[0, 2].set_ylabel('Number of Customers')
    axes[0, 2].set_title('Loyalty Changes Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Loyalty trajectories (first 10 customers)
    for i in range(min(10, len(results['loyalty_trajectories']))):
        trajectory = results['loyalty_trajectories'][i]
        axes[1, 0].plot(range(len(trajectory)), trajectory, alpha=0.6, linewidth=1)
    axes[1, 0].axhline(y=loyalty_threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold ({loyalty_threshold})')
    axes[1, 0].set_xlabel('Interaction Step')
    axes[1, 0].set_ylabel('Loyalty Score')
    axes[1, 0].set_title('Loyalty Trajectories (First 10 Customers)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Revenue distribution
    all_revenue = results['loyal_customers']['revenue'] + results['non_loyal_customers']['revenue']
    axes[1, 1].hist(all_revenue, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(all_revenue), color='red', linestyle='--', 
                      label=f'Average: ${np.mean(all_revenue):.2f}')
    axes[1, 1].set_xlabel('Revenue per Customer ($)')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Revenue Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Business metrics summary
    axes[1, 2].axis('off')
    loyal_avg = np.mean(loyal_revenue) if loyal_revenue else 0
    non_loyal_avg = np.mean(non_loyal_revenue) if non_loyal_revenue else 0
    
    summary_text = f"""
    === BUSINESS SUMMARY ===
    Total Customers: {len(loyal_revenue) + len(non_loyal_revenue)}
    Total Revenue: ${results['total_revenue']:.2f}
    
    Loyal Customers: {len(loyal_revenue)}
    - Average Revenue: ${loyal_avg:.2f}
    - % of Total: {len(loyal_revenue)/(len(loyal_revenue) + len(non_loyal_revenue))*100:.1f}%
    
    Non-Loyal Customers: {len(non_loyal_revenue)}
    - Average Revenue: ${non_loyal_avg:.2f}
    - % of Total: {len(non_loyal_revenue)/(len(loyal_revenue) + len(non_loyal_revenue))*100:.1f}%
    
    Revenue Gap: ${abs(loyal_avg - non_loyal_avg):.2f}
    
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
    plt.savefig('detailed_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    agent, env = load_trained_agent()
    if agent is None:
        return
    
    # Analyze single customer in detail
    analyze_single_customer_detailed(agent, env, loyalty_threshold=0.5)
    
    # Analyze multiple customers
    results = analyze_customer_journeys(agent, env, num_customers=200, loyalty_threshold=0.5)
    
    # Create visualizations
    visualize_detailed_results(results, loyalty_threshold=0.5)

if __name__ == "__main__":
    main()