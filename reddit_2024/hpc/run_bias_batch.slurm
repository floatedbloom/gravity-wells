#!/bin/bash
#SBATCH --job-name=bias_run
#SBATCH --partition=all
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=hpc/bias_run_%j.out
#SBATCH --error=hpc/bias_run_%j.err

set -euo pipefail

# 1) Project dir
cd ~/gravity_well/reddit_2024

# 2) Activate venv
if [ -f "$HOME/.venvs/project1/bin/activate" ]; then
  source "$HOME/.venvs/project1/bin/activate"
else
  echo "Venv not found at $HOME/.venvs/project1; proceeding with system Python"
fi

# 3) Base env
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS="$HOME/.ollama/models"
export PATH="$HOME/ollama/bin:$PATH"

echo "Starting Ollama on each node and verifying readiness..."
# 4) Start Ollama on each node and verify readiness
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -lc '
  set -e
  export PATH=$HOME/ollama/bin:$PATH
  export OLLAMA_HOST=127.0.0.1:11434
  pgrep -f "ollama serve" >/dev/null || (nohup ollama serve > "$HOME/ollama_$(hostname).log" 2>&1 & sleep 3)
  for i in {1..30}; do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
      echo "OK $(hostname)"; exit 0
    fi
    sleep 2
  done
  echo "FAIL $(hostname)"
  tail -n 200 "$HOME/ollama_$(hostname).log" || true
  exit 1
'

# 5) Cache dir (job-scoped)
CACHE_DIR="$(pwd)/cache/hpc_bias_cache_${SLURM_JOB_ID}"
mkdir -p "$CACHE_DIR"
echo "Cache directory: $CACHE_DIR"

echo "Launching analysis with per-task logs in hpc/run_${SLURM_JOB_ID}_%t.{out,err}"
# 6) Run analysis across tasks; per-task logs go to hpc/run_${SLURM_JOB_ID}_%t.{out,err}
srun --ntasks=4 --ntasks-per-node=1 \
  --export=ALL,OLLAMA_HOST,OLLAMA_MODELS \
  --output "$(pwd)/hpc/run_${SLURM_JOB_ID}_%t.out" \
  --error  "$(pwd)/hpc/run_${SLURM_JOB_ID}_%t.err" \
  python -u hpc/hpc_parallel_bias.py \
    --workers 16 \
    --sample-size 25 \
    --all

echo "Job completed. Check logs: hpc/bias_run_${SLURM_JOB_ID}.out|.err and hpc/run_${SLURM_JOB_ID}_*.{out,err}"


