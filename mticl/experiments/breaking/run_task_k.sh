#!/usr/bin/env bash
# Run maze planner ICL for the task_k / K-goals matrix. Execute from anywhere.
# Requires: cwd mticl/mticl for demos/; uses absolute --icl_config.log_path per run.
# If pyrallis fails to parse --task_goals, use per-run YAML configs instead (see comments).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTICL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS="$SCRIPT_DIR/results/task_k"

cd "$MTICL_ROOT"

planner_run() {
  local out_dir="$1"
  shift
  mkdir -p "$out_dir"
  python script/planner_icl.py \
    --icl_config.task AntMaze_UMazeDense-v3 \
    --icl_config.constraint_type Maze \
    --epochs 5 \
    --icl_config.log_path "$out_dir" \
    "$@"
}

# K=1 (single-task): maze_task matches the only goal in task_goals
planner_run "$RESULTS/k1_goal_0" \
  --maze_task 0 --task_goals '[0]' --exp_name k1_goal_0

planner_run "$RESULTS/k1_goal_3" \
  --maze_task 3 --task_goals '[3]' --exp_name k1_goal_3

planner_run "$RESULTS/k1_goal_7" \
  --maze_task 7 --task_goals '[7]' --exp_name k1_goal_7

# K>=2 multi-task: maze_task -1
planner_run "$RESULTS/k2_goals_0_1" \
  --maze_task -1 --task_goals '[0,1]' --exp_name k2_goals_0_1

planner_run "$RESULTS/k2_goals_0_5" \
  --maze_task -1 --task_goals '[0,5]' --exp_name k2_goals_0_5

planner_run "$RESULTS/k3_goals_0_3_7" \
  --maze_task -1 --task_goals '[0,3,7]' --exp_name k3_goals_0_3_7

planner_run "$RESULTS/k5_goals_0_2_4_6_8" \
  --maze_task -1 --task_goals '[0,2,4,6,8]' --exp_name k5_goals_0_2_4_6_8

planner_run "$RESULTS/k10_baseline" \
  --maze_task -1 --task_goals '[0,1,2,3,4,5,6,7,8,9]' --exp_name k10_baseline

echo "All runs finished. Outputs under $RESULTS"
