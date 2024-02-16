#!/bin/bash


echo "Submitting SLURM job for training transformer V1"

# Define the SLURM job parameters
JOB_NAME="train_transformer_V1"
OUTPUT_FILE="/gpfs/commons/groups/gursoy_lab/mstoll/codes/logs/Slurm_launch/Outputs/train_transformer_V1.txt"
ERROR_FILE="/gpfs/commons/groups/gursoy_lab/mstoll/codes/logs/Slurm_launch/Errors/train_transformer_V1.txt"

# Submit the SLURM job
sbatch --job-name="$JOB_NAME" \
        --partition=pe2 \
        --mail-type=ALL \
        --mail-user=mstoll@nygenome.org \
        --nodes=1 \
        --cpus-per-task=20 \
        --mem=80G \
        --time=20:00:00 \
        --output="$OUTPUT_FILE" \
        --error="$ERROR_FILE" \
        --wrap="/gpfs/commons/home/mstoll/anaconda3/envs/phewas/bin/python /gpfs/commons/groups/gursoy_lab/mstoll/codes/training/logistic_regression/train_logistic_regression.py"

echo "Submitted SLURM job for training transformer V1"


echo "Batch script completed"
