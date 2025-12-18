#!/bin/bash
#SBATCH --job-name=bias_analysis
#SBATCH --output=bias_analysis_%j.out
#SBATCH --error=bias_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=all

# HPC Bias Analysis SLURM Job Script
# ===================================
#
# This script submits the bias analysis job to the WSU HPC cluster.
# 
# Usage:
#   sbatch hpc_bias_analysis.slurm --subreddits progressive republicans mlb
#   sbatch hpc_bias_analysis.slurm --all --sample-size 50
#
# Job Configuration:
#   - 4 nodes (32 cores each = 128 total cores)
#   - 120GB memory per node
#   - 1 GPU per node (for potential future GPU acceleration)
#   - 24-hour time limit
#   - MPI distribution across nodes

# Load required modules (if available)
if command -v module &> /dev/null; then
    module load python/3.9
    module load mpi/openmpi-4.1.1
else
    echo "Module system not available, using system Python and MPI"
fi

# Activate project virtual environment (recommended)
# Priority: 1) Already-activated $VIRTUAL_ENV, 2) $GW_VENV_PATH override, 3) default path
if [ -n "${VIRTUAL_ENV}" ]; then
    VENV_PATH="${VIRTUAL_ENV}"
elif [ -n "${GW_VENV_PATH}" ]; then
    VENV_PATH="${GW_VENV_PATH}"
else
    VENV_PATH="${HOME}/.venvs/gw-hpc"
fi

if [ -d "${VENV_PATH}" ]; then
    echo "Activating venv at ${VENV_PATH}"
    # If user provided VENV_PATH points at the bin dir, adjust to root
    if [ -x "${VENV_PATH}/activate" ] && [ ! -x "${VENV_PATH}/bin/activate" ]; then
        # Looks like they passed .../bin; use its parent
        VENV_PATH="$(dirname "${VENV_PATH}")"
    fi
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
else
    echo "Venv not found at ${VENV_PATH}. Creating it now..."
    python3 -m venv "${VENV_PATH}"
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip
    # Install only the minimal set needed for HPC run
    pip install -r "$(pwd)/../requirements-minimal.txt"
fi

# Use the venv's python explicitly everywhere
PYTHON_EXEC="${VENV_PATH}/bin/python"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# Use cache directory in project folder (must be created manually)
CACHE_DIR="$(pwd)/cache/hpc_bias_cache_${SLURM_JOB_ID}"

# Parse command line arguments
SUBREDDITS=""
SAMPLE_SIZE=25
WORKERS=16
ALL_SUBREDDITS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --subreddits)
            shift
            SUBREDDITS="$1"
            shift
            ;;
        --all)
            ALL_SUBREDDITS=true
            shift
            ;;
        --sample-size)
            shift
            SAMPLE_SIZE="$1"
            shift
            ;;
        --workers)
            shift
            WORKERS="$1"
            shift
            ;;
        --estimate-only)
            ESTIMATE_ONLY="--estimate-only"
            shift
            ;;
        --clear-progress)
            CLEAR_PROGRESS="--clear-progress"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command (export key env vars to all ranks)
CMD="mpirun -np ${SLURM_NNODES} \
    -x OLLAMA_HOST -x OLLAMA_MODELS -x PYTHONPATH -x OMP_NUM_THREADS \
    ${PYTHON_EXEC} hpc_parallel_bias.py"
CMD="${CMD} --workers ${WORKERS}"
CMD="${CMD} --sample-size ${SAMPLE_SIZE}"
CMD="${CMD} --cache-dir ${CACHE_DIR}"

if [ "$ALL_SUBREDDITS" = true ]; then
    CMD="${CMD} --all"
elif [ -n "$SUBREDDITS" ]; then
    CMD="${CMD} --subreddits ${SUBREDDITS}"
fi

if [ -n "$ESTIMATE_ONLY" ]; then
    CMD="${CMD} ${ESTIMATE_ONLY}"
fi

if [ -n "$CLEAR_PROGRESS" ]; then
    CMD="${CMD} ${CLEAR_PROGRESS}"
fi

# Print job information
echo "=========================================="
echo "HPC Bias Analysis Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "Cores per node: ${SLURM_CPUS_PER_TASK}"
echo "Memory per node: ${SLURM_MEM_PER_NODE}MB"
echo "Time limit: ${SLURM_TIME_LIMIT}"
echo "Cache directory: ${CACHE_DIR}"
echo "Command: ${CMD}"
echo "=========================================="

# Check if Ollama is available
if ! command -v ollama &> /dev/null; then
    echo "Warning: Ollama not found. The script will attempt to use alternative methods."
    echo "For best results, ensure Ollama is installed and running."
fi

# Ensure required Python packages are available in the venv
"${PYTHON_EXEC}" -c "import mpi4py, ollama, numpy" 2>/dev/null || {
    echo "Installing required Python packages into venv..."
    pip install mpi4py ollama numpy
}

# Start Ollama on each compute node if not already running
echo "Starting Ollama on compute nodes..."
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
    # Check if Ollama is already running on this node
    if ! pgrep -f "ollama serve" > /dev/null; then
        echo "Node $(hostname): Starting Ollama server..."
        export OLLAMA_HOST=127.0.0.1:11434
        nohup ollama serve > ollama_$(hostname).log 2>&1 &
        sleep 5  # Give Ollama time to start
        
        # Wait for Ollama to be ready
        for i in {1..30}; do
            if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
                echo "Node $(hostname): Ollama is ready"
                break
            fi
            echo "Node $(hostname): Waiting for Ollama to start... ($i/30)"
            sleep 2
        done
        
        # Set OLLAMA_MODELS to shared directory if home is shared
        if [ -d "$HOME/.ollama/models" ]; then
            export OLLAMA_MODELS="$HOME/.ollama/models"
            echo "Node $(hostname): Using shared model directory: $OLLAMA_MODELS"
        fi
        
        # Check if model is available
        if ! ollama list | grep -q "qwen2.5:7b"; then
            echo "Node $(hostname): qwen2.5:7b not found"
            # Try to pull if we have network access
            if curl -s --connect-timeout 5 https://ollama.ai > /dev/null 2>&1; then
                echo "Node $(hostname): Attempting to pull model..."
                ollama pull qwen2.5:7b
            else
                echo "Node $(hostname): No internet access, cannot pull model"
                echo "Node $(hostname): Model must be available in shared storage"
            fi
        else
            echo "Node $(hostname): qwen2.5:7b found"
        fi
    else
        echo "Node $(hostname): Ollama already running"
    fi
'

echo "Ollama startup completed on all nodes"

# Test Ollama connectivity from each node
echo "Testing Ollama connectivity from each node..."
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
    echo "Node $(hostname): Testing Ollama connection..."
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Node $(hostname): ✓ Ollama HTTP accessible"
        if command -v ollama &> /dev/null; then
            if ollama list > /dev/null 2>&1; then
                echo "Node $(hostname): ✓ Ollama CLI accessible"
                echo "Node $(hostname): Available models:"
                ollama list || true
            else
                echo "Node $(hostname): ✗ Ollama CLI not accessible"
            fi
        else
            echo "Node $(hostname): ✗ Ollama command not found"
        fi
    else
        echo "Node $(hostname): ✗ Ollama HTTP not accessible"
    fi
'

# Set environment variables for the analysis
export OLLAMA_HOST=127.0.0.1:11434
if [ -d "$HOME/.ollama/models" ]; then
    export OLLAMA_MODELS="$HOME/.ollama/models"
fi

# Run the analysis
echo "Starting bias analysis at $(date)"
start_time=$(date +%s)

${CMD}

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "=========================================="
echo "Analysis completed at $(date)"
echo "Total runtime: ${duration} seconds ($(echo "scale=2; ${duration}/3600" | awk '{printf "%.2f", $1}') hours)"
echo "=========================================="

# Clean up cache directory (optional - comment out to keep cache)
# echo "Cleaning up cache directory..."
# rm -rf ${CACHE_DIR}

echo "Job completed successfully!"
