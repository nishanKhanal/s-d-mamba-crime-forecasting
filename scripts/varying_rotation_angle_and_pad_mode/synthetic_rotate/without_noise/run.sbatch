#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train-smamba-crime
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --output=logs/sbatch/r_syn_varying_a_and_pm_wo_noise.%j
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
bash /mnt/home/khanalni/shield-project/extra/S-D-Mamba/scripts/varying_rotation_angle_and_pad_mode/synthetic_rotate/without_noise/S_Mamba.sh
