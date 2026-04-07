#!/usr/bin/env bash
# Breaking Experiment 3: observation noise on the constraint classifier.
# Adds Gaussian noise (sigma sweep) to batches fed into constraint training.
# Run from anywhere; requires bash. Execute: bash experiments/breaking/run_obs_noise.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTICL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/obs_noise"

cd "$MTICL_ROOT"

mkdir -p "$RESULTS_DIR/sigma_0"
mkdir -p "$RESULTS_DIR/sigma_001"
mkdir -p "$RESULTS_DIR/sigma_005"
mkdir -p "$RESULTS_DIR/sigma_01"
mkdir -p "$RESULTS_DIR/sigma_02"

COMMON=(
  python script/planner_icl.py
  --icl_config.task AntMaze_UMazeDense-v3
  --icl_config.constraint_type Maze
  --epochs 5
  --maze_task -1
)

echo "=== MT-ICL: sigma=0.0 (clean baseline) ==="
"${COMMON[@]}" \
  --exp_name obs_noise_sigma_0 \
  --obs_noise_std 0.0 \
  --icl_config.log_path "$RESULTS_DIR/sigma_0"

echo "=== MT-ICL: sigma=0.01 ==="
"${COMMON[@]}" \
  --exp_name obs_noise_sigma_001 \
  --obs_noise_std 0.01 \
  --icl_config.log_path "$RESULTS_DIR/sigma_001"

echo "=== MT-ICL: sigma=0.05 ==="
"${COMMON[@]}" \
  --exp_name obs_noise_sigma_005 \
  --obs_noise_std 0.05 \
  --icl_config.log_path "$RESULTS_DIR/sigma_005"

echo "=== MT-ICL: sigma=0.1 ==="
"${COMMON[@]}" \
  --exp_name obs_noise_sigma_01 \
  --obs_noise_std 0.1 \
  --icl_config.log_path "$RESULTS_DIR/sigma_01"

echo "=== MT-ICL: sigma=0.2 ==="
"${COMMON[@]}" \
  --exp_name obs_noise_sigma_02 \
  --obs_noise_std 0.2 \
  --icl_config.log_path "$RESULTS_DIR/sigma_02"

echo "All runs finished. Logs under $RESULTS_DIR"
