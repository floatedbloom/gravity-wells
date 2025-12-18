#!/bin/bash
# HPC Setup Script for Bias Analysis
# ===================================
#
# This script sets up the environment for running bias analysis on the WSU HPC cluster.
#
# Usage:
#   ./setup_hpc.sh
#   source setup_hpc.sh

echo "Setting up HPC environment for bias analysis..."

# Check if we're on the HPC cluster
if [[ "$HOSTNAME" == *"tricities"* ]] || [[ "$SLURM_JOB_ID" != "" ]]; then
    echo "Detected HPC environment"
    
    # Load required modules
    module load python/3.9
    module load mpi/openmpi-4.1.1
    
    # Install required Python packages
    echo "Installing required Python packages..."
    pip install --user mpi4py ollama numpy pandas scipy scikit-learn
    
    # Create cache directory
    mkdir -p /tmp/hpc_bias_cache
    
    # Set environment variables
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export OMP_NUM_THREADS=1
    
    echo "HPC environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Ensure Ollama is running: ollama serve"
    echo "2. Submit a job: sbatch hpc_bias_analysis.slurm --subreddits progressive republicans mlb"
    echo "3. Monitor job: squeue -u $USER"
    echo "4. Check logs: tail -f bias_analysis_<JOB_ID>.out"
    
else
    echo "Not on HPC cluster. This script is designed for the WSU HPC environment."
    echo "Please run this on the HPC cluster: ssh username@vcluster.tricities.wsu.edu"
fi
