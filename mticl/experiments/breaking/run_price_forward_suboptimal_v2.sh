#!/usr/bin/env bash
# PRICE-forward v2 (K=10): all five conditions (clean, vr5, vr10, vr20, vr50).
# Same flags as run_price_forward_suboptimal.sh; results under price_forward_suboptimal_v2/
set -euo pipefail
cd "$(dirname "$0")/../.."

RESULT_ROOT="experiments/breaking/results/price_forward_suboptimal_v2"
COMMON=(
  --icl_config.task AntMaze_UMazeDense-v3
  --icl_config.constraint_type Maze
  --icl_config.seed 100
  --icl_config.suffix breaking_subopt_price_forward_v2
  --exp_name breaking_subopt_price_forward_v2
  --maze_task -1
  --epochs 5
  --use_price
  --icl_config.expert_batch_fraction 0.25
  --icl_config.price.route_all_dispreferred true
  --icl_config.price.routed_batch_fraction 1.0
  --icl_config.price.pairs_per_constraint_phase 256
  --icl_config.price.routed_positions_cap 16384
)

declare -A TIMES
declare -A EXHAUSTIONS

run_one() {
  local cond="$1"
  shift
  local outdir="${RESULT_ROOT}/${cond}"
  mkdir -p "${outdir}"

  local start_t
  start_t=$(date +%s)

  python script/planner_icl.py "${COMMON[@]}" "$@" \
    --icl_config.log_path "${outdir}" 2>&1 | tee "${outdir}/run.log"

  local end_t
  end_t=$(date +%s)
  TIMES["${cond}"]=$(( end_t - start_t ))

  # Count retry exhaustions from the log
  local exh
  exh=$(grep -c "WARNING: gen_valid_demo failed" "${outdir}/run.log" 2>/dev/null || echo 0)
  EXHAUSTIONS["${cond}"]="${exh}"
}

# --- Run all five conditions ---
run_one clean

for suf in _vr5 _vr10 _vr20 _vr50; do
  tag="${suf#_}"
  run_one "${tag}" --demo_dir demos/corrupted --demo_suffix "${suf}"
done

# --- Summary table ---
echo ""
echo "================================================================="
echo "  PRICE-forward v2 — Summary"
echo "================================================================="
printf "%-10s  %10s  %12s  %10s\n" "Condition" "Final IoU" "Exhaustions" "Time (s)"
echo "-----------------------------------------------------------------"

for cond in clean vr5 vr10 vr20 vr50; do
  outdir="${RESULT_ROOT}/${cond}"
  iou=$(python -c "
import sys; sys.path.insert(0, '.')
from experiments.breaking.compare_iou_utils import iou_for_run_dir
v = iou_for_run_dir('${outdir}')
print(f'{v:.4f}' if v is not None else 'N/A')
" 2>/dev/null || echo "N/A")
  printf "%-10s  %10s  %12s  %10s\n" \
    "${cond}" "${iou}" "${EXHAUSTIONS[${cond}]:-?}" "${TIMES[${cond}]:-?}"
done

echo "================================================================="
echo "Results saved under ${RESULT_ROOT}"
