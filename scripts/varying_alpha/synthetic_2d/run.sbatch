#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train-smamba-synthetic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --output=logs/sbatch/syn2d_varying_alpha.%j
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
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/varying_alpha/synthetic_2d/S_Mamba.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/varying_alpha/synthetic_2d/iTransformer.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/varying_alpha/synthetic_2d/Informer.sh
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/varying_alpha/synthetic_2d/Transformer.sh