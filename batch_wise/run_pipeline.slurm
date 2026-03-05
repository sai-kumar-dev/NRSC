#!/bin/bash
#SBATCH --job-name=era5_pinn_pipeline
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=errors/%x_%j.err

set -euo pipefail

mkdir -p logs
mkdir -p errors

echo "================================="
echo "ERA5 PINN PIPELINE JOB START"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================="

# Activate conda
conda activate era5_pinn

echo "Environment: era5_pinn"
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"

# Optional: control thread usage

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Threads: $OMP_NUM_THREADS"

echo "Running pipeline..."

python run_pipeline.py 
--sampler hybrid 
--batch_size 1000 
--batches 2
--start_year 1990 
--end_year 2025

echo "================================="
echo "JOB FINISHED"
echo "End Time: $(date)"
echo "================================="
