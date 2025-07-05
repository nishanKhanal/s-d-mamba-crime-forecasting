#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train-traffic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --output=logs/sbatch/traffic-s-I.%j
#SBATCH --mem=64G
#SBATCH --partition=gpu


# Load necessary modules (adjust as needed for your system)
ml miniconda3
ml cuda/11.8

# echo running conda init
# conda init 
# echo ran conda init

source ~/.bashrc
# Activate a specific conda environment 
conda activate $smamba_env

# python run.py
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/Traffic/S_Mamba.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/Traffic/Informer.sh
# bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/Traffic/iTransformer.sh
# bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/Traffic/Transformer.sh