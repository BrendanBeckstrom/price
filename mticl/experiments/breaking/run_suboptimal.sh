#!/usr/bin/env bash
# Breaking Experiment 2: suboptimal expert demos (violation-rate sweep) + MT-ICL.
# Run from anywhere; requires bash. Execute: bash experiments/breaking/run_suboptimal.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTICL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results/suboptimal"
CORRUPT_DIR="demos/corrupted"

cd "$MTICL_ROOT"

mkdir -p "$RESULTS_DIR"
mkdir -p "$CORRUPT_DIR"

echo "=== Generating corrupted demos at 5%, 10%, 20%, 50% violation rates ==="
python experiments/breaking/gen_corrupted_demos.py --violation_rate 0.05 --output_dir "$CORRUPT_DIR" --seed 42
python experiments/breaking/gen_corrupted_demos.py --violation_rate 0.10 --output_dir "$CORRUPT_DIR" --seed 42
python experiments/breaking/gen_corrupted_demos.py --violation_rate 0.20 --output_dir "$CORRUPT_DIR" --seed 42
python experiments/breaking/gen_corrupted_demos.py --violation_rate 0.50 --output_dir "$CORRUPT_DIR" --seed 42

COMMON=(
  python script/planner_icl.py
  --icl_config.task AntMaze_UMazeDense-v3
  --icl_config.constraint_type Maze
  --epochs 5
  --maze_task -1
)

echo "=== MT-ICL: clean baseline (default demos/) ==="
mkdir -p "$RESULTS_DIR/suboptimal_clean"
"${COMMON[@]}" \
  --exp_name suboptimal_clean \
  --icl_config.log_path "$RESULTS_DIR/suboptimal_clean"

echo "=== MT-ICL: corrupted demos (suffix _vr5 / _vr10 / _vr20 / _vr50) ==="
mkdir -p "$RESULTS_DIR/suboptimal_vr5"
"${COMMON[@]}" \
  --exp_name suboptimal_vr5 \
  --demo_dir "$CORRUPT_DIR" \
  --demo_suffix _vr5 \
  --icl_config.log_path "$RESULTS_DIR/suboptimal_vr5"

mkdir -p "$RESULTS_DIR/suboptimal_vr10"
"${COMMON[@]}" \
  --exp_name suboptimal_vr10 \
  --demo_dir "$CORRUPT_DIR" \
  --demo_suffix _vr10 \
  --icl_config.log_path "$RESULTS_DIR/suboptimal_vr10"

mkdir -p "$RESULTS_DIR/suboptimal_vr20"
"${COMMON[@]}" \
  --exp_name suboptimal_vr20 \
  --demo_dir "$CORRUPT_DIR" \
  --demo_suffix _vr20 \
  --icl_config.log_path "$RESULTS_DIR/suboptimal_vr20"

mkdir -p "$RESULTS_DIR/suboptimal_vr50"
"${COMMON[@]}" \
  --exp_name suboptimal_vr50 \
  --demo_dir "$CORRUPT_DIR" \
  --demo_suffix _vr50 \
  --icl_config.log_path "$RESULTS_DIR/suboptimal_vr50"

echo "All runs finished. Logs under $RESULTS_DIR"
