#!/usr/bin/env bash
# PRICE (K=10) on the same corrupted demo tree as Experiment 2; outputs for IoU vs MT-ICL plots.
# Does not modify experiments/breaking/results/suboptimal/ (MT-ICL baselines).
set -euo pipefail
cd "$(dirname "$0")/../.."

RESULT_ROOT="experiments/breaking/results/price_suboptimal"
COMMON=(
  --icl_config.task AntMaze_UMazeDense-v3
  --icl_config.constraint_type Maze
  --icl_config.seed 100
  --icl_config.suffix breaking_subopt_price
  --exp_name breaking_subopt_price
  --maze_task -1
  --epochs 5
  --use_price
)

run_one() {
  local cond="$1"
  shift
  python script/planner_icl.py "${COMMON[@]}" "$@" \
    --icl_config.log_path "${RESULT_ROOT}/${cond}"
}

mkdir -p "${RESULT_ROOT}/clean"
run_one clean

for suf in _vr5 _vr10 _vr20 _vr50; do
  tag="${suf#_}"
  mkdir -p "${RESULT_ROOT}/${tag}"
  run_one "${tag}" --demo_dir demos/corrupted --demo_suffix "${suf}"
done

echo "PRICE suboptimal runs finished under ${RESULT_ROOT}"
