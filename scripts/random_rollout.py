"""Quick smoke test for the customer offer environment using a random policy."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2, help="number of episodes to roll out")
    parser.add_argument("--seed", type=int, default=7, help="random seed for reproducibility")
    args = parser.parse_args()

    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        print(f"\nEpisode {episode + 1}")
        for t in range(env.config.horizon):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(
                f"t={t:02d} action={action} accepted={info['accepted']} "
                f"revenue={reward:6.2f} loyalty={info['loyalty_score']:.2f} "
                f"recency={info['recency_of_purchase']}"
            )
            if terminated or truncated:
                break
        print(f"Total episode revenue: {total_reward:.2f}")


if __name__ == "__main__":
    main()
