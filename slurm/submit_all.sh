#!/bin/bash
# Submit Track A + Track B to SLURM on musica-inn
# Full-node jobs: 4 tasks per node, array index = node index

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Generate task files
python generate_tasks.py

# Create logs directory
mkdir -p ~/TurboGNN_Benchmark/logs

cd ~/TurboGNN_Benchmark

# Track A: 72 tasks / 4 GPUs per node = 18 nodes
NODES_A=$(( (72 + 3) / 4 ))  # ceil(72/4) = 18
echo "=== Submitting Track A (72 tasks on $NODES_A nodes: 4 datasets x 6 graphs x 3 seeds, HVG=200) ==="
JOB_A=$(sbatch --array=0-$((NODES_A - 1)) --parsable "$SCRIPT_DIR/job_benchmark.sh" "$SCRIPT_DIR/tasks_track_a.txt")
echo "Track A job array: $JOB_A (array 0-$((NODES_A - 1)))"

# Track B: 120 tasks / 4 GPUs per node = 30 nodes
NODES_B=$(( (120 + 3) / 4 ))  # ceil(120/4) = 30
echo "=== Submitting Track B (120 tasks on $NODES_B nodes: 4 datasets x 5 graphs x 2 HVG x 3 seeds) ==="
JOB_B=$(sbatch --array=0-$((NODES_B - 1)) --parsable "$SCRIPT_DIR/job_benchmark.sh" "$SCRIPT_DIR/tasks_track_b.txt")
echo "Track B job array: $JOB_B (array 0-$((NODES_B - 1)))"

echo ""
echo "Total: 192 tasks on $((NODES_A + NODES_B)) nodes (48 nodes)"
echo "Monitor: squeue -u \$USER"
echo "Results: ~/TurboGNN_Benchmark/results_benchmark/hvg{200,500,1000}/"
