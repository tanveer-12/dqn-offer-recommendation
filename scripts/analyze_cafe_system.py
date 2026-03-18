"""Analysis of cafe-based recommendation system with loyalty threshold and detailed metrics."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # For cluster compatibility
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv
from offer_env.dqn_agent import DQNAgent


def load_trained_agent():
    """Load the trained cafe recommendation agent."""
    env = CustomerOfferEnv(CustomerConfig(), seed=42)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    
    try:
        agent.qnetwork_local.load_state_dict(torch.load('cafe_recommendation_model.pth'))
        print("Cafe recommendation model loaded successfully!")
        return agent, env
    except FileNotFoundError:
        print("No trained model found. Please run training first.")
        return None, None


def analyze_system_performance(agent, env, num_customers=10000, loyalty_threshold=0.5):
    """Analyze the trained system's performance."""
    print(f"\nAnalyzing cafe system performance on {num_customers} customers...")
    print(f"Loyalty threshold: {loyalty_threshold}")
    
    results = {
        'loyal_customers': {'revenue': [], 'loyalty_changes': [], 'offer_dist': {i: 0 for i in range(5)}},
        'not_loyal_customers': {'revenue': [], 'loyalty_changes': [], 'offer_dist': {i: 0 for i in range(5)}},
        'offer_usage': {i: 0 for i in range(5)},
        'offer_acceptance': {i: {'accepted': 0, 'rejected': 0} for i in range(5)},
        'all_revenue': [],
        'loyalty_trajectories': [],
        'customer_types': {'loyal': 0, 'not_loyal': 0}
    }
    
    for customer_id in range(num_customers):
        state, _ = env.reset(seed=42 + customer_id)
        initial_loyalty = state[0]
        customer_revenue = 0.0
        customer_loyalty_trajectory = [initial_loyalty]
        
        # Simulate customer journey
        for t in range(12):  # 12 monthly visits
            action = agent.act(state, eps=0.0)  # No exploration in analysis
            
            # Track offer usage and acceptance
            results['offer_usage'][action] += 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info['accepted']:
                results['offer_acceptance'][action]['accepted'] += 1
            else:
                results['offer_acceptance'][action]['rejected'] += 1
            
            customer_revenue += reward
            customer_loyalty_trajectory.append(info['loyalty_score'])
            
            state = next_state
            if done:
                break
        
        # Classify customer and store results
        final_loyalty = customer_loyalty_trajectory[-1]
        if final_loyalty > loyalty_threshold:
            results['loyal_customers']['revenue'].append(customer_revenue)
            results['customer_types']['loyal'] += 1
        else:
            results['not_loyal_customers']['revenue'].append(customer_revenue)
            results['customer_types']['not_loyal'] += 1
        
        results['all_revenue'].append(customer_revenue)
        results['loyalty_trajectories'].append(customer_loyalty_trajectory)
    
    return results


def analyze_detailed_customer_journeys(agent, env, loyalty_threshold=0.5):
    """Analyze detailed journeys for sample customers."""
    print(f"\n{'='*80}")
    print("DETAILED CUSTOMER JOURNEY ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze different customer types
    customer_types = [
        ("High-Loyalty Customer", [0.8, 100, -1, 0, 1]),
        ("Low-Loyalty Customer", [0.2, 40, -1, 0, 5]),
        ("Medium-Loyalty Customer", [0.5, 60, -1, 0, 2])
    ]
    
    for customer_name, initial_state in customer_types:
        print(f"\n{customer_name} Journey:")
        print("Step | Action | Accepted | Revenue | Loyalty | Recency")
        print("-" * 55)
        
        state, _ = env.reset(seed=1000)  # Reset to get proper environment state
        total_revenue = 0.0
        
        for t in range(12):  # 12 monthly visits
            action = agent.act(state, eps=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"{t:4d} | {action:6d} | {str(info['accepted']):8s} | {reward:7.2f} | {info['loyalty_score']:7.2f} | {info['recency_of_purchase']:7d}")
            
            total_revenue += reward
            state = next_state
            if done:
                break
        
        final_loyalty = info['loyalty_score']
        customer_type = "LOYAL" if final_loyalty > loyalty_threshold else "NOT LOYAL"
        
        print("-" * 55)
        print(f"Total Revenue: ${total_revenue:.2f}")
        print(f"Final Loyalty: {final_loyalty:.2f}")
        print(f"Customer Type: {customer_type}")


def create_analysis_plots(results, loyalty_threshold):
    """Create meaningful analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Revenue comparison: loyal vs not loyal
    loyal_revenue = results['loyal_customers']['revenue']
    not_loyal_revenue = results['not_loyal_customers']['revenue']
    
    axes[0, 0].boxplot([loyal_revenue, not_loyal_revenue], labels=['Loyal', 'Not Loyal'])
    axes[0, 0].set_ylabel('Revenue per Customer ($)')
    axes[0, 0].set_title('Revenue: Loyal vs Not Loyal Customers')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Offer acceptance rates
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    acceptance_rates = []
    for i in range(5):
        total = results['offer_acceptance'][i]['accepted'] + results['offer_acceptance'][i]['rejected']
        rate = (results['offer_acceptance'][i]['accepted'] / total * 100) if total > 0 else 0
        acceptance_rates.append(rate)
    
    bars = axes[0, 1].bar(offer_names, acceptance_rates, 
                          color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    axes[0, 1].set_xlabel('Offer Type')
    axes[0, 1].set_ylabel('Acceptance Rate (%)')
    axes[0, 1].set_title('Offer Acceptance Rates')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, rate in zip(bars, acceptance_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Revenue distribution
    avg_revenue = np.mean(results['all_revenue'])
    axes[0, 2].hist(results['all_revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 2].axvline(avg_revenue, color='red', linestyle='--', 
                       label=f'Average: ${avg_revenue:.2f}')
    axes[0, 2].set_xlabel('Revenue per Customer ($)')
    axes[0, 2].set_ylabel('Number of Customers')
    axes[0, 2].set_title('Revenue Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Loyalty trajectories (first 10 customers)
    for i in range(min(10, len(results['loyalty_trajectories']))):
        trajectory = results['loyalty_trajectories'][i]
        axes[1, 0].plot(range(len(trajectory)), trajectory, alpha=0.6, linewidth=1)
    axes[1, 0].axhline(y=loyalty_threshold, color='red', linestyle='--', alpha=0.8, 
                       label=f'Threshold ({loyalty_threshold})')
    axes[1, 0].set_xlabel('Monthly Visit')
    axes[1, 0].set_ylabel('Loyalty Score')
    axes[1, 0].set_title('Loyalty Trajectories (First 10 Customers)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Offer usage distribution
    offer_counts = [results['offer_usage'][i] for i in range(5)]
    bars = axes[1, 1].bar(offer_names, offer_counts, 
                         color=['blue', 'orange', 'green', 'red', 'purple'], alpha=0.7)
    axes[1, 1].set_xlabel('Offer Type')
    axes[1, 1].set_ylabel('Number of Times Used')
    axes[1, 1].set_title('Offer Usage Distribution')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add count labels
    for bar, count in zip(bars, offer_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 6. Customer type breakdown
    labels = ['Loyal', 'Not Loyal']
    sizes = [results['customer_types']['loyal'], results['customer_types']['not_loyal']]
    colors = ['green', 'red']
    axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Customer Type Distribution')
    
    plt.tight_layout()
    plt.savefig('cafe_system_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nANALYSIS PLOTS SAVED AS: cafe_system_analysis.png")


def print_comprehensive_analysis(results, loyalty_threshold):
    """Print comprehensive analysis results."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SYSTEM ANALYSIS")
    print(f"{'='*80}")
    
    # Basic statistics
    total_customers = len(results['all_revenue'])
    total_revenue = sum(results['all_revenue'])
    avg_revenue = np.mean(results['all_revenue'])
    
    print(f"\nBASIC STATISTICS:")
    print(f"Total customers analyzed: {total_customers}")
    print(f"Total revenue generated: ${total_revenue:.2f}")
    print(f"Average revenue per customer: ${avg_revenue:.2f}")
    
    # Customer classification
    loyal_count = results['customer_types']['loyal']
    not_loyal_count = results['customer_types']['not_loyal']
    
    print(f"\nCUSTOMER CLASSIFICATION:")
    print(f"Loyal customers (loyalty > {loyalty_threshold}): {loyal_count} ({loyal_count/total_customers*100:.1f}%)")
    print(f"Not loyal customers: {not_loyal_count} ({not_loyal_count/total_customers*100:.1f}%)")
    
    # Revenue by customer type
    loyal_avg = np.mean(results['loyal_customers']['revenue']) if results['loyal_customers']['revenue'] else 0
    not_loyal_avg = np.mean(results['not_loyal_customers']['revenue']) if results['not_loyal_customers']['revenue'] else 0
    
    print(f"\nREVENUE BY CUSTOMER TYPE:")
    print(f"Average revenue - Loyal customers: ${loyal_avg:.2f}")
    print(f"Average revenue - Not loyal customers: ${not_loyal_avg:.2f}")
    print(f"Revenue gap: ${abs(loyal_avg - not_loyal_avg):.2f}")
    
    # Offer analysis
    print(f"\nOFFER USAGE ANALYSIS:")
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    for i, name in enumerate(offer_names):
        total = results['offer_acceptance'][i]['accepted'] + results['offer_acceptance'][i]['rejected']
        acceptance_rate = (results['offer_acceptance'][i]['accepted'] / total * 100) if total > 0 else 0
        print(f"{name:12s}: {total:4d} uses | {results['offer_acceptance'][i]['accepted']:3d} accepted | {acceptance_rate:5.1f}% rate")
    
    # Cafe-specific insights
    avg_monthly_spending = avg_revenue / 12  # 12 monthly visits
    print(f"\nCAFE BUSINESS INSIGHTS:")
    print(f"Average monthly spending per customer: ${avg_revenue:.2f}")
    print(f"Average daily spending per customer: ${avg_monthly_spending:.2f}")
    print(f"Potential monthly revenue from {total_customers} customers: ${total_revenue:.2f}")
    print(f"Revenue per loyal customer: ${loyal_avg:.2f}")
    print(f"Revenue per not loyal customer: ${not_loyal_avg:.2f}")


def main():
    agent, env = load_trained_agent()
    if agent is None:
        return
    
    # Analyze detailed customer journeys
    analyze_detailed_customer_journeys(agent, env, loyalty_threshold=0.5)
    
    # Analyze system performance
    results = analyze_system_performance(agent, env, num_customers=10000, loyalty_threshold=0.5)
    
    # Create analysis plots
    create_analysis_plots(results, loyalty_threshold=0.5)
    
    # Print comprehensive analysis
    print_comprehensive_analysis(results, loyalty_threshold=0.5)


if __name__ == "__main__":
    main()