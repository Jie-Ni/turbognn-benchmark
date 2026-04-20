#!/bin/bash
#SBATCH --job-name=turbognn
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# LNZ version: uses /scratch/fs201130 instead of /data/fs201130

set -euo pipefail

DATA=/scratch/fs201130/jn20658
TASK_FILE="${1:?Usage: sbatch --array=0-N job_split_lnz.sh <task_file>}"
TASK_FILE_ABS=$(realpath "$TASK_FILE")
ARRAY_ID=${SLURM_ARRAY_TASK_ID}
TOTAL_TASKS=$(wc -l < "$TASK_FILE_ABS")

echo "=== Node: $(hostname), Array ID: $ARRAY_ID ==="
nvidia-smi -L 2>/dev/null | head -4

eval "$($DATA/miniconda3/bin/conda shell.bash hook)"
conda activate $DATA/envs/turbognn

WORK_DIR=$DATA/TurboGNN_Benchmark
cd "$WORK_DIR"
export TURBOGNN_DATA_DIR=$DATA/TurboGNN/data

PIDS=()
for GPU_IDX in 0 1 2 3; do
    TASK_IDX=$((ARRAY_ID * 4 + GPU_IDX))
    if [ "$TASK_IDX" -ge "$TOTAL_TASKS" ]; then
        continue
    fi

    LINE=$(sed -n "$((TASK_IDX + 1))p" "$TASK_FILE_ABS")
    DATASET=$(echo "$LINE" | awk '{print $1}')
    GRAPH=$(echo "$LINE" | awk '{print $2}')
    NUM_HVG=$(echo "$LINE" | awk '{print $3}')
    SEED=$(echo "$LINE" | awk '{print $4}')
    FOLD_START=$(echo "$LINE" | awk '{print $5}')
    FOLD_END=$(echo "$LINE" | awk '{print $6}')

    RESULTS_DIR=$WORK_DIR/results_benchmark/hvg${NUM_HVG}
    mkdir -p "$RESULTS_DIR" "$WORK_DIR/logs"

    echo "GPU $GPU_IDX: $DATASET $GRAPH hvg=$NUM_HVG seed=$SEED folds=$FOLD_START-$FOLD_END"

    CUDA_VISIBLE_DEVICES=$GPU_IDX PYTHONUNBUFFERED=1 python run_benchmark.py \
        --full \
        --datasets "$DATASET" \
        --graph-types "$GRAPH" \
        --num-hvg "$NUM_HVG" \
        --seed "$SEED" \
        --fold-start "$FOLD_START" \
        --fold-end "$FOLD_END" \
        --results-dir "$RESULTS_DIR" \
        > "$WORK_DIR/logs/split_${TASK_IDX}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} tasks, waiting..."
FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL + 1))
done
echo "=== Done ($FAIL failures) at $(date) ==="
