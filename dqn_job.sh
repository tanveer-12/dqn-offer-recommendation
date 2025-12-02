#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -J "dqn3"
#SBATCH -o dqn-balanced-%j.out
#SBATCH -e dqn-balanced-%j.err
#SBATCH -p academic
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

module load python/3.10.12
module load miniconda3
module load cuda

cd /home/tfnu/RL-PersonalizedOfferRecommendation

source /home/tfnu/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run extended training with 10,000 episodes
echo "Starting extended training with 10,000 episodes..."
python scripts/train_balanced.py --episodes 10000 


# After training, run analysis scripts
echo "Training completed. Running analysis..."
python scripts/analyze_balanced.py
python scripts/visualize_agent_behavior.py

echo "All analysis completed!"