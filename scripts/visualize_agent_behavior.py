"""Visualize how the trained agent behaves with different customer states."""

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
    """Load the trained agent for visualization."""
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

def get_q_values_for_state(agent, state):
    """Get Q-values for all actions for a given state."""
    state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
    with torch.no_grad():
        q_values = agent.qnetwork_local(state_tensor)
    return q_values.numpy().flatten()

def visualize_state_action_preferences(agent):
    """Visualize how the agent's preferences change with different customer states."""
    print("\n=== STATE-ACTION PREFERENCE ANALYSIS ===")
    
    # Define different customer types to analyze
    customer_profiles = [
        # [loyalty, spend, last_offer, accepted, recency]
        ([0.9, 120, -1, 0, 0], "High-value, loyal customer"),
        ([0.2, 40, -1, 0, 8], "Low-value, inactive customer"), 
        ([0.6, 80, 2, 0, 5], "Moderate customer, recently rejected 10% discount"),
        ([0.4, 100, -1, 0, 2], "High spender, low loyalty"),
        ([0.8, 60, 4, 1, 0], "Loyal customer, just accepted premium offer"),
    ]
    
    offer_names = ['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (state, description) in enumerate(customer_profiles):
        if i < len(axes):
            q_values = get_q_values_for_state(agent, state)
            
            # Create bar chart for this customer type
            bars = axes[i].bar(offer_names, q_values, 
                              color=['blue', 'orange', 'green', 'red', 'purple'], 
                              alpha=0.7)
            
            # Color bars based on Q-value magnitude
            colors = plt.cm.viridis(q_values / max(q_values + 1e-8))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[i].set_title(f'{description}\nState: L={state[0]:.1f}, S=${state[1]:.0f}, R={state[4]}')
            axes[i].set_ylabel('Q-Value (Expected Future Revenue)')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(axes) > len(customer_profiles):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('agent_state_action_preferences.png', dpi=300, bbox_inches='tight')
    plt.show()

def simulate_customer_journey(agent, env, initial_state, customer_name="Sample Customer"):
    """Simulate a complete customer journey to see how the agent adapts."""
    print(f"\n=== SIMULATING {customer_name.upper()} JOURNEY ===")
    
    # Reset environment and set initial state (simplified - you might need to modify env)
    state, _ = env.reset(seed=42)
    print(f"Starting state: {state}")
    
    journey_data = {
        'step': [],
        'state': [],
        'action': [],
        'reward': [],
        'acceptance': [],
        'total_revenue': []
    }
    
    total_revenue = 0
    
    for step in range(15):  # Simulate 15 interactions
        action = agent.act(state, eps=0.0)  # No exploration, pure strategy
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_revenue += reward
        
        print(f"Step {step}: State={state[:3]} (Loyalty={state[0]:.2f}, Spend={state[1]:.0f}, Recency={state[4]})")
        print(f"  -> Action: {action} ({['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium'][action]})")
        print(f"  -> Revenue: ${reward:.2f}, Accepted: {info['accepted']}, Loyalty: {info['loyalty_score']:.2f}")
        print(f"  -> Total Revenue So Far: ${total_revenue:.2f}")
        print()
        
        # Store journey data
        journey_data['step'].append(step)
        journey_data['state'].append(state.copy())
        journey_data['action'].append(action)
        journey_data['reward'].append(reward)
        journey_data['acceptance'].append(info['accepted'])
        journey_data['total_revenue'].append(total_revenue)
        
        state = next_state
        if done:
            break
    
    return journey_data

def visualize_customer_journey(journey_data, customer_name="Customer"):
    """Visualize the customer journey data."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Revenue over time
    axes[0, 0].plot(journey_data['step'], journey_data['total_revenue'], 
                   'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Interaction Step')
    axes[0, 0].set_ylabel('Cumulative Revenue ($)')
    axes[0, 0].set_title(f'{customer_name}: Cumulative Revenue Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Actions taken over time
    actions = journey_data['action']
    action_names = [f'{i}\n{["No", "5%", "10%", "BOGO", "Prem"][i]}' for i in actions]
    axes[0, 1].bar(journey_data['step'], actions, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Interaction Step')
    axes[0, 1].set_ylabel('Action Taken')
    axes[0, 1].set_title(f'{customer_name}: Offers Recommended Over Time')
    axes[0, 1].set_yticks(range(5))
    axes[0, 1].set_yticklabels(['No Offer', '5% Disc', '10% Disc', 'BOGO', 'Premium'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Acceptance rate over time
    acceptance = [1 if acc else 0 for acc in journey_data['acceptance']]
    axes[1, 0].bar(journey_data['step'], acceptance, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Interaction Step')
    axes[1, 0].set_ylabel('Offer Accepted (1=Yes, 0=No)')
    axes[1, 0].set_title(f'{customer_name}: Offer Acceptance Over Time')
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['No', 'Yes'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loyalty score over time (if available in state)
    loyalty_scores = [state[0] for state in journey_data['state']]
    axes[1, 1].plot(journey_data['step'], loyalty_scores, 'r-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Interaction Step')
    axes[1, 1].set_ylabel('Loyalty Score')
    axes[1, 1].set_title(f'{customer_name}: Loyalty Score Over Time')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{customer_name.lower()}_journey_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    agent, env = load_trained_agent()
    if agent is None:
        return
    
    # Visualize state-action preferences
    visualize_state_action_preferences(agent)
    
    # Simulate different customer journeys
    test_states = [
        ([0.8, 100, -1, 0, 1], "High-Value Customer"),
        ([0.3, 50, -1, 0, 8], "Low-Value Customer"),
        ([0.6, 75, 2, 0, 5], "Moderate Customer - Recently Rejected"),
    ]
    
    for state, name in test_states:
        journey_data = simulate_customer_journey(agent, env, state, name)
        visualize_customer_journey(journey_data, name)

if __name__ == "__main__":
    main()