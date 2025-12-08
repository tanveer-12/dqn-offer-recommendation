"""Greedy evaluation of a trained DQN agent on the customer offer environment."""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv  # noqa: E402
from agents import DQNAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000, help="number of eval episodes")
    parser.add_argument("--seed", type=int, default=123, help="base seed for reproducibility")
    parser.add_argument("--model-path", type=str, default="models/dqn.pth", help="path to saved model (.pth)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden size used during training")
    parser.add_argument("--double-dqn", action="store_true", help="use Double DQN architecture (match training)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return parser.parse_args()


def greedy_eval(args: argparse.Namespace) -> None:
    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    agent = DQNAgent(
        state_dim=obs.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=args.hidden_dim,
        double_dqn=args.double_dqn,
        device=args.device,
    )
    agent.load(args.model_path)
    agent.epsilon = 0.0  # greedy

    rewards: List[float] = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        total = 0.0
        for _ in range(env.config.horizon):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)

    print(
        f"Greedy eval over {args.episodes} episodes "
        f"| mean={np.mean(rewards):.2f} | std={np.std(rewards):.2f} "
        f"| min={np.min(rewards):.2f} | max={np.max(rewards):.2f}"
    )


def main() -> None:
    args = parse_args()
    greedy_eval(args)


if __name__ == "__main__":
    main()
