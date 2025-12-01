#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -J "dqn"
#SBATCH -o dqn%j.out
#SBATCH -e dqn%j.err
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
python scripts/train_dqn.py --episodes 10000 --max_t 100 --eps_decay 0.997 --eps_end 0.01


# After training, run analysis scripts
echo "Training completed. Running analysis..."
python scripts/analyze_results.py
python scripts/visualize_agent_behavior.py

echo "All analysis completed!"