#!/bin/bash


echo "Submitting SLURM job for training transformer V1"

# Define the SLURM job parameters
JOB_NAME="get_data_tree_counts"
OUTPUT_FILE="/gpfs/commons/groups/gursoy_lab/mstoll/codes/logs/Slurm_launch/Outputs/train_transformer_V1.txt"
ERROR_FILE="/gpfs/commons/groups/gursoy_lab/mstoll/codes/logs/Slurm_launch/Errors/train_transformer_V1.txt"

# Submit the SLURM job
sbatch --job-name="$JOB_NAME" \
        --partition=pe2 \
        --mail-type=ALL \
        --mail-user=mstoll@nygenome.org \
        --nodes=1 \
        --cpus-per-task=2\
        --mem=300G \
        --time=20:00:00 \
        --output="$OUTPUT_FILE" \
        --error="$ERROR_FILE" \
        --wrap="/gpfs/commons/home/mstoll/anaconda3/envs/phewas/bin/python /gpfs/commons/groups/gursoy_lab/mstoll/codes/models/data_form/load_pheno_file_counts.py"

echo "Submitted SLURM job for get scores"


echo "Batch script completed"
