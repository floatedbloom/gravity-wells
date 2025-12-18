#!/bin/bash
# HPC Analysis Launcher
# =====================
#
# This script launches HPC bias analysis from the main reddit_2024 directory.
# It automatically changes to the hpc directory and runs the analysis.
#
# Usage:
#   ./run_hpc_analysis.sh --subreddits progressive republicans mlb
#   ./run_hpc_analysis.sh --all --sample-size 50

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$SCRIPT_DIR/hpc"

# Check if HPC directory exists
if [ ! -d "$HPC_DIR" ]; then
    echo "Error: HPC directory not found at $HPC_DIR"
    exit 1
fi

# Change to HPC directory and run the analysis
cd "$HPC_DIR"

# Pass all arguments to the SLURM script
echo "Launching HPC bias analysis from $HPC_DIR"
echo "Arguments: $@"
echo ""

# Run the SLURM script with all arguments
sbatch hpc_bias_analysis.slurm "$@"

echo ""
echo "Job submitted! Monitor with: squeue -u $USER"
