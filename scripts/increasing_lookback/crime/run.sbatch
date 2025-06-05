#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train-smamba-crime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --output=logs/sbatch/crime_inc_lookback_all.%j
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
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/Informer_M.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/Informer.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/iTransformer.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/Reformer_M.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/Reformer.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/S_Mamba.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/Transformer_M.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/increasing_lookback/crime/Transformer.sh