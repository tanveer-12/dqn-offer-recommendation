"""Plot training curves from the CSV produced by train_dqn.py."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=str, help="path to metrics CSV from train_dqn.py")
    parser.add_argument("--show", action="store_true", help="display the plot instead of just saving")
    parser.add_argument("--out", type=str, default="training_curve.png", help="output plot file")
    return parser.parse_args()


def load_metrics(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        episodes, rewards, rolling, losses = [], [], [], []
        acceptance = []
        action_cols = [c for c in reader.fieldnames if c.startswith("action_")]
        action_counts = {c: [] for c in action_cols}
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            # rolling column name encodes window length; take the first matching key
            roll_keys = [k for k in row.keys() if k.startswith("rolling_reward_")]
            rolling.append(float(row[roll_keys[0]]))
            losses.append(float(row["mean_loss"]))
            acceptance.append(float(row["acceptance_rate"]))
            for c in action_cols:
                action_counts[c].append(int(row[c]))
    return {
        "episodes": episodes,
        "rewards": rewards,
        "rolling": rolling,
        "losses": losses,
        "acceptance": acceptance,
        "action_counts": action_counts,
    }


def plot(metrics: dict, out_path: Path, show: bool) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(metrics["episodes"], metrics["rewards"], alpha=0.3, label="reward")
    axes[0].plot(metrics["episodes"], metrics["rolling"], linewidth=2, label="rolling reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(metrics["episodes"], metrics["losses"], color="tab:red", alpha=0.7)
    axes[1].set_ylabel("Mean loss")

    axes[2].plot(metrics["episodes"], metrics["acceptance"], color="tab:green", alpha=0.7, label="acceptance rate")
    for name, counts in metrics["action_counts"].items():
        axes[2].plot(metrics["episodes"], counts, alpha=0.3, label=name)
    axes[2].set_ylabel("Acceptance / action counts")
    axes[2].set_xlabel("Episode")
    axes[2].legend(ncol=3, fontsize="small")

    plt.tight_layout()
    fig.savefig(out_path)
    if show:
        plt.show()
    print(f"Saved plot to {out_path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    metrics = load_metrics(csv_path)
    plot(metrics, Path(args.out), args.show)


if __name__ == "__main__":
    main()
