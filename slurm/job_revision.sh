#!/bin/bash
#SBATCH --job-name=turbognn_revision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Canonical runner for task rows emitted by generate_revision_tasks.py.
# Usage: sbatch --array=0-N slurm/job_revision.sh TASK_FILE SCHEDULE_MANIFEST

set -euo pipefail

TASK_FILE=$(realpath "${1:?Pass a generate_revision_tasks.py task file}")
SCHEDULE_MANIFEST=$(realpath "${2:?Pass its generate_revision_tasks.py manifest}")
WORK_DIR=$(realpath "${TURBOGNN_WORK_DIR:?Set TURBOGNN_WORK_DIR to the clean Git checkout}")
DATA_ROOT=$(realpath "${TURBOGNN_DATA_ROOT:?Set TURBOGNN_DATA_ROOT}")
DATASET_PASSPORT=$(realpath "${TURBOGNN_DATASET_PASSPORT:?Set TURBOGNN_DATASET_PASSPORT}")
CONTROL_MAP=$(realpath "${TURBOGNN_CONTROL_MAP:?Set TURBOGNN_CONTROL_MAP}")
TARGET_MAP=$(realpath "${TURBOGNN_TARGET_MAP:?Set TURBOGNN_TARGET_MAP}")
PREPROCESSING_CONFIG=$(realpath "${TURBOGNN_PREPROCESSING_CONFIG:?Set TURBOGNN_PREPROCESSING_CONFIG}")
MODEL_CONFIG=$(realpath "${TURBOGNN_MODEL_CONFIG:?Set TURBOGNN_MODEL_CONFIG}")
ENVIRONMENT_LOCK=$(realpath "${TURBOGNN_ENVIRONMENT_LOCK:?Set TURBOGNN_ENVIRONMENT_LOCK}")
GRAPH_SOURCE_CONFIG=$(realpath "${TURBOGNN_GRAPH_SOURCE_CONFIG:?Set TURBOGNN_GRAPH_SOURCE_CONFIG}")
GRAPH_ENSEMBLE_CONFIG=$(realpath "${TURBOGNN_GRAPH_ENSEMBLE_CONFIG:?Set TURBOGNN_GRAPH_ENSEMBLE_CONFIG}")
CODE_COMMIT="${TURBOGNN_CODE_COMMIT:?Set TURBOGNN_CODE_COMMIT to the exact release SHA}"
RESULTS_ROOT="${TURBOGNN_RESULTS_ROOT:?Set TURBOGNN_RESULTS_ROOT outside the checkout}"
mkdir -p "$RESULTS_ROOT"
RESULTS_ROOT=$(realpath "$RESULTS_ROOT")
case "$RESULTS_ROOT/" in
    "$WORK_DIR"/*) echo "TURBOGNN_RESULTS_ROOT must be outside TURBOGNN_WORK_DIR" >&2; exit 2 ;;
esac

TOTAL_TASKS=$(python "$WORK_DIR/slurm/validate_revision_launch.py" \
    --manifest "$SCHEDULE_MANIFEST" --task-file "$TASK_FILE")
ARRAY_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array}"
EXPECTED_ARRAY_TASKS=$(((TOTAL_TASKS + 3) / 4))
EXPECTED_ARRAY_MAX=$((EXPECTED_ARRAY_TASKS - 1))
if [ "${SLURM_ARRAY_TASK_COUNT:?Missing SLURM_ARRAY_TASK_COUNT}" -ne "$EXPECTED_ARRAY_TASKS" ] || \
   [ "${SLURM_ARRAY_TASK_MIN:?Missing SLURM_ARRAY_TASK_MIN}" -ne 0 ] || \
   [ "${SLURM_ARRAY_TASK_MAX:?Missing SLURM_ARRAY_TASK_MAX}" -ne "$EXPECTED_ARRAY_MAX" ] || \
   [ "${SLURM_ARRAY_TASK_STEP:?Missing SLURM_ARRAY_TASK_STEP}" -ne 1 ]; then
    echo "SLURM array must exactly cover 0-${EXPECTED_ARRAY_MAX} with step 1" >&2
    exit 2
fi
if [ "$ARRAY_ID" -lt 0 ] || [ "$ARRAY_ID" -gt "$EXPECTED_ARRAY_MAX" ]; then
    echo "SLURM_ARRAY_TASK_ID $ARRAY_ID is outside 0-${EXPECTED_ARRAY_MAX}" >&2
    exit 2
fi
mkdir -p "$RESULTS_ROOT/logs"
cd "$WORK_DIR"

PIDS=()
for GPU_IDX in 0 1 2 3; do
    TASK_IDX=$((ARRAY_ID * 4 + GPU_IDX))
    if [ "$TASK_IDX" -ge "$TOTAL_TASKS" ]; then
        continue
    fi
    LINE=$(sed -n "$((TASK_IDX + 1))p" "$TASK_FILE")
    read -r STAGE DATASET GRAPH NUM_HVG SEED PANEL_SIZE FOLD_START FOLD_END GRAPH_SEED CONFIG_HASH EXTRA <<< "$LINE"
    if [ -n "${EXTRA:-}" ] || [ -z "${CONFIG_HASH:-}" ]; then
        echo "Malformed task row $TASK_IDX: expected exactly 10 fields" >&2
        exit 2
    fi
    GRAPH_INSTANCE_ARGS=()
    if [ "$GRAPH_SEED" != "-" ]; then
        GRAPH_INSTANCE_ARGS=(--graph-instance-seed "$GRAPH_SEED")
    fi
    RESULTS_DIR="$RESULTS_ROOT/$STAGE/hvg${NUM_HVG}"
    mkdir -p "$RESULTS_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_IDX" PYTHONUNBUFFERED=1 python run_benchmark.py \
        --stage "$STAGE" \
        --datasets "$DATASET" \
        --graph-types "$GRAPH" \
        --num-hvg "$NUM_HVG" \
        --seed "$SEED" \
        --panel-size "$PANEL_SIZE" \
        --fold-start "$FOLD_START" \
        --fold-end "$FOLD_END" \
        --dataset-passport "$DATASET_PASSPORT" \
        --data-root "$DATA_ROOT" \
        --control-map "$CONTROL_MAP" \
        --target-map "$TARGET_MAP" \
        --preprocessing-config "$PREPROCESSING_CONFIG" \
        --model-config "$MODEL_CONFIG" \
        --code-commit "$CODE_COMMIT" \
        --environment-lock "$ENVIRONMENT_LOCK" \
        --schedule-manifest "$SCHEDULE_MANIFEST" \
        --schedule-config-hash "$CONFIG_HASH" \
        --graph-source-config "$GRAPH_SOURCE_CONFIG" \
        --graph-ensemble-config "$GRAPH_ENSEMBLE_CONFIG" \
        "${GRAPH_INSTANCE_ARGS[@]}" \
        --results-dir "$RESULTS_DIR" \
        > "$RESULTS_ROOT/logs/task_${SLURM_ARRAY_JOB_ID}_${TASK_IDX}.log" 2>&1 &
    PIDS+=("$!")
done

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "Array element $ARRAY_ID launched no tasks" >&2
    exit 2
fi

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=$((FAIL + 1))
done
if [ "$FAIL" -ne 0 ]; then
    echo "$FAIL revision task(s) failed; inspect logs and terminal manifests" >&2
    exit 1
fi
