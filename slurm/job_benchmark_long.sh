#!/bin/bash
#SBATCH --job-name=turbognn
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Long-running version for Replogle datasets (2000+ conditions)
# 24-hour wall time to handle 50 folds with large datasets

set -euo pipefail

DATA=/data/fs201130/jn20658
TASK_FILE="${1:?Usage: sbatch --array=0-N job_benchmark_long.sh <task_file>}"
TASK_FILE_ABS=$(realpath "$TASK_FILE")
ARRAY_ID=${SLURM_ARRAY_TASK_ID}
TOTAL_TASKS=$(wc -l < "$TASK_FILE_ABS")

echo "=== Node: $(hostname), Array ID: $ARRAY_ID ==="
echo "Time: $(date)"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi unavailable"

eval "$($DATA/miniconda3/bin/conda shell.bash hook)"
conda activate $DATA/envs/turbognn

WORK_DIR=$DATA/TurboGNN_Benchmark
cd "$WORK_DIR"
export TURBOGNN_DATA_DIR=$DATA/TurboGNN/data

PIDS=()
for GPU_IDX in 0 1 2 3; do
    TASK_IDX=$((ARRAY_ID * 4 + GPU_IDX))
    if [ "$TASK_IDX" -ge "$TOTAL_TASKS" ]; then
        echo "GPU $GPU_IDX: No task (index $TASK_IDX >= $TOTAL_TASKS), idle"
        continue
    fi

    LINE=$(sed -n "$((TASK_IDX + 1))p" "$TASK_FILE_ABS")
    DATASET=$(echo "$LINE" | awk '{print $1}')
    GRAPH=$(echo "$LINE" | awk '{print $2}')
    NUM_HVG=$(echo "$LINE" | awk '{print $3}')
    SEED=$(echo "$LINE" | awk '{print $4}')

    RESULTS_DIR=$WORK_DIR/results_benchmark/hvg${NUM_HVG}
    mkdir -p "$RESULTS_DIR" "$WORK_DIR/logs"

    echo "GPU $GPU_IDX: task=$TASK_IDX dataset=$DATASET graph=$GRAPH hvg=$NUM_HVG seed=$SEED"

    CUDA_VISIBLE_DEVICES=$GPU_IDX python run_benchmark.py \
        --full \
        --datasets "$DATASET" \
        --graph-types "$GRAPH" \
        --num-hvg "$NUM_HVG" \
        --seed "$SEED" \
        --results-dir "$RESULTS_DIR" \
        > "$WORK_DIR/logs/replogle_task_${TASK_IDX}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} tasks, waiting..."

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL + 1))
done

if [ "$FAIL" -gt 0 ]; then
    echo "WARNING: $FAIL task(s) failed"
fi

echo "=== All tasks completed at $(date) ==="
