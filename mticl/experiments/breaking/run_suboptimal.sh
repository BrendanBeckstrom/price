#!/usr/bin/env bash
# Breaking Experiment 2: MT-ICL (K=10) on clean vs behaviorally corrupted maze demos.
# Writes under experiments/breaking/results/suboptimal/<condition>/ (do not delete if baselining).
set -euo pipefail
cd "$(dirname "$0")/../.."

RESULT_ROOT="experiments/breaking/results/suboptimal"
COMMON=(
  --icl_config.task AntMaze_UMazeDense-v3
  --icl_config.constraint_type Maze
  --icl_config.seed 100
  --icl_config.suffix breaking_subopt_mticl
  --exp_name breaking_subopt_mticl
  --maze_task -1
  --epochs 5
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

echo "MT-ICL suboptimal runs finished under ${RESULT_ROOT}"
