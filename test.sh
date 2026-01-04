#!/bin/bash
#
# --- SBATCH 配置 (与之前相同) ---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/test.out
#SBATCH -e logs/test.err
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=512G

set -x

# --- 1. 激活环境 ---
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate trlt
