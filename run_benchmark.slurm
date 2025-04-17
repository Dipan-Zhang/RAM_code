#!/bin/bash
#SBATCH --job-name=ram_benchmark
#SBATCH --output=ram_benchmark_%j.out
#SBATCH --error=ram_benchmark_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --time=00:60:00

# ---- User Input ----
TASK=${1:-all}  # Default to 'all' if no argument is passed

# ---- Validate Input ----
if [[ "$TASK" != "all" && "$TASK" != "portable" && "$TASK" != "articulate" ]]; then
    echo "Invalid task type: $TASK"
    echo "Usage: sbatch run_ram_benchmark.slurm [all|portable|articulate]"
    exit 1
fi

# ---- Load environment ----
module load cuda/11.8.0
source ~/software/miniforget3/etc/profile.d/conda.sh
conda activate RAM

# ---- Run Benchmark ----
# cd code/RAM_code/
echo "Running RAM benchmark for task: $TASK"
srun python run_benchmark.py -t "$TASK" -n 5 --same_trial_name

echo "Job completed for task: $TASK"
