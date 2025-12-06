Reinforcement Learning for Personalized Offer Recommendation
Team Members: Tanveer Kaur, Davud Azizov

Environment Setup (code scaffold only, no RL agent yet)
- Prerequisites: Python 3.10+ recommended.
- Create and activate a virtual environment (example): `python3 -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Quick smoke test of the simulated environment (random actions only): `python scripts/random_rollout.py --episodes 1`

1. Introduction and Motivation
Personalized recommendations are at the heart of modern digital marketing systems used by companies like Amazon, Starbucks, and Netflix. Traditional recommendation systems focus on maximizing short-term engagement (e.g., click-through rate), but they fail to optimize for long-term user value — such as repeat purchases, loyalty, and profitability.
Our project aims to design a Reinforcement Learning (RL)-based offer recommendation system that learns to recommend deals or discounts which maximize long-term cumulative profit rather than just immediate sales.
Instead of relying on static customer data, the RL agent will interact with a simulated customer environment, learning over time which offers lead to sustained engagement and revenue growth.
2. Problem Definition and Goals
Problem Statement:
How can an RL agent dynamically recommend personalized offers to customers to maximize long-term profit and retention?
Goals:
● Develop a customer behavior simulator to model how different customers respond to offers.
● Implement a reinforcement learning agent (e.g., Q-learning or DQN) that learns optimal offer policies through interaction.
● Compare RL-based recommendations with heuristic or rule-based strategies.
● Evaluate success using metrics such as average cumulative reward, profit per customer, and customer retention rate.
3. Methodology
Environment Design
We will create a custom Gymnasium-style environment that simulates customers. Each episode represents one customer over several time steps (e.g., daily or weekly interactions).
● State Representation: [loyalty_score, average_spend, last_offer, accepted_last_offer, recency_of_purchase]
● Action Space: a0: No offer a1: 5% discount a2: 10% discount a3: Buy-1-Get-1 deal a4: Premium membership offer
Learning Algorithm
We will begin with Tabular Q-Learning for discrete state spaces, and extend to Deep Q-Networks (DQN) if needed for better generalization.
Baseline Comparison
We will implement:
1. Heuristic Model: Fixed rules (e.g., always offer 10% discount if last offer failed).
2. Contextual Bandit: Optimizes immediate reward only.
3. RL Agent: Optimizes cumulative long-term reward.
5. Expected Outcomes
By the end of the project, we expect to:
● Develop a working RL agent that learns optimal discount policies.
● Show that the RL agent outperforms heuristic baselines in terms of average cumulative reward and profitability.
● Visualize the agent’s learning progress and decision patterns.
● Optionally, create a small interactive demo (e.g., using Streamlit) showcasing the recommendation process.(It is an optional step if we got time at the end of the project, we could present a visual demo)
6. Timeline and Deliverables
In the first week, we will focus on defining the environment design and logic. This includes specifying the customer state representation, action space, and reward function. By the end of this stage, we expect to have a working simulation environment that models basic customer behavior in response to different offers.
During the second week, we will implement the core reinforcement learning algorithm using the “TBD” approach. We will run initial experiments to verify that the agent is learning and that the reward signal behaves as expected. The deliverable for this week will be a functioning RL training loop and initial learning curves showing improvement over episodes.
In the third week, we plan to implement and benchmark heuristic and contextual bandit baselines. These models will serve as comparison points to evaluate how much the RL approach improves over simpler strategies. The main deliverable will be a set of quantitative results comparing average rewards, profits, and customer retention rates.
The fourth week will focus on refining and tuning the RL model, improving performance stability, and generating visualizations to illustrate the learning progress. At this stage, we will produce plots of cumulative rewards and profit trends, as well as interpret how the agent’s offer selection policy evolves over time. A draft of the report will also be prepared during this period.
In the final week, we will complete the integration and polishing phase. This will include cleaning the codebase, preparing visual outputs, and finalizing the report. If time permits, we will also develop a simple interactive demo—either a console-based or Streamlit visualization—to showcase the recommendation process in action. By the end of this week, the full project deliverables will include the working code implementation, analysis plots, and the final report summarizing the design, experiments, and findings.
7. Feasibility and Challenges
The project is computationally lightweight since the environment is simulated and non-visual. The main challenges include:
● Designing a realistic yet tractable customer simulator.
● Properly shaping the reward to reflect long-term goals.
● Avoiding overfitting to a single customer type (possible extension: multi-customer simulation).
Both tasks are feasible on standard personal computers or university machines, with expected training times under a few hours.
8. Expected Deliverables at Project End
● Full Python codebase with modular environment and RL agents
● Plots and metrics comparing RL vs. heuristic baselines
● A short final report summarizing design, results, and insights
● (Optional) A visual demo showing offer recommendations in action

Training a simple DQN agent (baseline)
- Install deps: `pip install -r requirements.txt`
- Run training (CPU ok for this env): `python scripts/train_dqn.py --episodes 400`
- Stability knobs: `--reward-scale 0.01` (default) keeps targets small; `--min-replay` warms up buffer; `--grad-clip` caps exploding gradients
- Optional: `--device cuda` if available, `--save-path models/dqn.pth` to persist weights (default saves to `models/dqn.pth`)
- Resume from checkpoint: `python scripts/train_dqn.py --episodes 200 --load-path models/dqn.pth --save-path models/dqn.pth`
- Rolling average: script prints rolling mean reward over the last N episodes (default 50); adjust via `--rolling-window`
- Model tweaks: `--double-dqn` (recommended) to reduce overestimation bias; `--hidden-dim 256` to try a wider MLP
- Keep checkpoints separate per architecture to avoid load errors:
  - Vanilla example: `--save-path models/dqn_vanilla_128.pth`
  - Double DQN example: `--save-path models/dqn_double_256.pth`
- Expect epsilon-greedy exploration that anneals over ~8k steps with a target network synced every 1k steps
- Greedy eval of a saved agent (epsilon=0): `python scripts/eval_dqn.py --model-path models/dqn.pth --episodes 50 --hidden-dim <train_hidden> [--double-dqn]`
- Log and plot training metrics:
  - Enable CSV logging: add `--log-csv logs/train_metrics.csv` to `train_dqn.py`
  - Plot rewards/loss/acceptance/actions: `python scripts/plot_training.py logs/train_metrics.csv --out training_curve.png`
