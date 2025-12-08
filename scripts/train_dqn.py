"""Train a DQN agent on the customer offer environment."""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from offer_env import CustomerConfig, CustomerOfferEnv  # noqa: E402
from agents import DQNAgent, Transition  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=400, help="number of training episodes")
    parser.add_argument("--seed", type=int, default=7, help="random seed for env and agent")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="mini-batch size")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="replay buffer capacity")
    parser.add_argument("--epsilon-decay", type=int, default=8_000, help="steps to anneal epsilon")
    parser.add_argument("--target-update", type=int, default=1_000, help="steps between target syncs")
    parser.add_argument("--min-replay", type=int, default=500, help="steps to collect before training")
    parser.add_argument("--reward-scale", type=float, default=0.01, help="multiply rewards by this factor")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="max grad norm (0 to disable)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden size for Q-network MLP")
    parser.add_argument("--double-dqn", action="store_true", help="enable Double DQN target calculation (recommended)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--load-path", type=str, default="", help="optional path to load a saved model")
    parser.add_argument("--save-path", type=str, default="models/dqn.pth", help="path to save the model (empty to skip)")
    parser.add_argument("--rolling-window", type=int, default=50, help="window size for rolling avg reward")
    parser.add_argument("--log-csv", type=str, default="", help="optional path to write per-episode metrics CSV")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    env = CustomerOfferEnv(CustomerConfig(), seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        min_replay_size=args.min_replay,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        hidden_dim=args.hidden_dim,
        double_dqn=args.double_dqn,
        device=args.device,
    )
    if args.load_path:
        agent.load(args.load_path)
        # Start near exploit after loading so epsilon doesn't reset to full exploration.
        agent.epsilon = agent.epsilon_end
        print(f"Loaded model from {args.load_path}")

    total_steps = 0
    rewards_history = []
    episode_rows: List[List[float]] = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_loss = []
        action_counts = np.zeros(action_dim, dtype=int)
        accept_count = 0
        for t in range(env.config.horizon):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            scaled_reward = reward * args.reward_scale
            agent.store(
                Transition(
                    state=state,
                    action=action,
                    reward=scaled_reward,
                    next_state=next_state,
                    done=done,
                )
            )
            loss = agent.train_step()
            if loss > 0.0:
                ep_loss.append(loss)

            state = next_state
            ep_reward += reward
            action_counts[action] += 1
            if info.get("accepted", False):
                accept_count += 1
            total_steps += 1
            if done:
                break

        mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        rewards_history.append(ep_reward)
        window = max(1, args.rolling_window)
        rolling_mean = float(np.mean(rewards_history[-window:]))
        acceptance_rate = float(accept_count) / float(max(1, np.sum(action_counts)))
        episode_rows.append(
            [
                ep + 1,
                ep_reward,
                rolling_mean,
                agent.epsilon,
                mean_loss,
                acceptance_rate,
                *action_counts.tolist(),
            ]
        )
        print(
            f"Episode {ep+1:04d} | steps={t+1:02d} | reward={ep_reward:7.2f} "
            f"| epsilon={agent.epsilon:.3f} | loss={mean_loss:.4f} | roll_mean({window})={rolling_mean:7.2f}"
        )

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.save_path)
        print(f"Saved model to {args.save_path}")

    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        header = [
            "episode",
            "reward",
            f"rolling_reward_{window}",
            "epsilon",
            "mean_loss",
            "acceptance_rate",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "action_4",
        ]
        with open(args.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(episode_rows)
        print(f"Wrote metrics to {args.log_csv}")


if __name__ == "__main__":
    main()
