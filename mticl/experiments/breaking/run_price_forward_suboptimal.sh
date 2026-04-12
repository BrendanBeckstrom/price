#!/usr/bin/env bash
# PRICE-forward (K=10): weaker expert batch mix + route_all_dispreferred + larger routed pool.
# Same demo tree as Experiment 2; writes under experiments/breaking/results/price_forward_suboptimal/
set -euo pipefail
cd "$(dirname "$0")/../.."

RESULT_ROOT="experiments/breaking/results/price_forward_suboptimal"
COMMON=(
  --icl_config.task AntMaze_UMazeDense-v3
  --icl_config.constraint_type Maze
  --icl_config.seed 100
  --icl_config.suffix breaking_subopt_price_forward
  --exp_name breaking_subopt_price_forward
  --maze_task -1
  --epochs 5
  --use_price
  --icl_config.expert_batch_fraction 0.25
  --icl_config.price.route_all_dispreferred true
  --icl_config.price.routed_batch_fraction 1.0
  --icl_config.price.pairs_per_constraint_phase 256
  --icl_config.price.routed_positions_cap 16384
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

echo "PRICE-forward suboptimal runs finished under ${RESULT_ROOT}"
