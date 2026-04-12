@echo off
REM ============================================================
REM  Verification + 3-method sweep for PRICE vs MT-ICL
REM  Run from Anaconda prompt in the repo root:
REM    run_verify_and_sweep.bat
REM ============================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM ----------------------------------------------------------
REM  Step 1: Import / syntax check (run from repo root where price/ lives)
REM ----------------------------------------------------------
echo === Step 1: Syntax check ===
python -c "from price.maze_metrics import wall_violation_count; from price.router import sample_trajectory_pairs; print('All imports OK')"
if errorlevel 1 (
    echo FAILED: import check. Fix errors before continuing.
    exit /b 1
)

cd mticl

REM ----------------------------------------------------------
REM  Step 2: Regression — clean maze, all defaults, no PRICE
REM ----------------------------------------------------------
echo.
echo === Step 2: Regression run (clean, no PRICE, 5 epochs) ===
mkdir experiments\breaking\results\regression_check 2>nul
python script/planner_icl.py ^
    --icl_config.task AntMaze_UMazeDense-v3 ^
    --icl_config.constraint_type Maze ^
    --icl_config.seed 100 ^
    --icl_config.suffix regression_check ^
    --exp_name regression_check ^
    --maze_task -1 ^
    --epochs 5 ^
    --icl_config.log_path experiments/breaking/results/regression_check
if errorlevel 1 (
    echo FAILED: regression run.
    exit /b 1
)
echo Regression run complete. Check IoU in experiments\breaking\results\regression_check\

REM ----------------------------------------------------------
REM  Step 3: Smoke test — PRICE-forward, 1 epoch
REM ----------------------------------------------------------
echo.
echo === Step 3: Smoke test (PRICE-forward, 1 epoch) ===
mkdir experiments\breaking\results\smoke_test 2>nul
python script/planner_icl.py ^
    --icl_config.task AntMaze_UMazeDense-v3 ^
    --icl_config.constraint_type Maze ^
    --icl_config.seed 100 ^
    --icl_config.suffix smoke_price_forward ^
    --exp_name smoke_price_forward ^
    --maze_task -1 ^
    --epochs 1 ^
    --use_price true ^
    --icl_config.expert_batch_fraction 0.25 ^
    --icl_config.price.route_all_dispreferred true ^
    --icl_config.price.routed_batch_fraction 1.0 ^
    --icl_config.price.pairs_per_constraint_phase 256 ^
    --icl_config.price.routed_positions_cap 16384 ^
    --icl_config.log_path experiments/breaking/results/smoke_test
if errorlevel 1 (
    echo FAILED: smoke test.
    exit /b 1
)
echo Smoke test complete. Look for [price] epoch log lines above.

REM ----------------------------------------------------------
REM  Step 4: Full sweep (MT-ICL baseline already ran — skipping)
REM ----------------------------------------------------------

REM ----------------------------------------------------------
REM  Step 4a: PRICE-conservative (defaults)
REM ----------------------------------------------------------
echo.
echo === Step 4b: PRICE-conservative (clean + corrupted) ===
set PRICE_CON_ROOT=experiments\breaking\results\price_suboptimal
for %%C in (clean vr5 vr10 vr20 vr50) do mkdir "!PRICE_CON_ROOT!\%%C" 2>nul

python script/planner_icl.py ^
    --icl_config.task AntMaze_UMazeDense-v3 ^
    --icl_config.constraint_type Maze ^
    --icl_config.seed 100 ^
    --icl_config.suffix breaking_subopt_price ^
    --exp_name breaking_subopt_price ^
    --maze_task -1 ^
    --epochs 5 ^
    --use_price true ^
    --icl_config.log_path %PRICE_CON_ROOT%/clean

for %%S in (_vr5 _vr10 _vr20 _vr50) do (
    set "TAG=%%S"
    set "TAG=!TAG:~1!"
    echo --- PRICE-conservative condition: !TAG! ---
    python script/planner_icl.py ^
        --icl_config.task AntMaze_UMazeDense-v3 ^
        --icl_config.constraint_type Maze ^
        --icl_config.seed 100 ^
        --icl_config.suffix breaking_subopt_price ^
        --exp_name breaking_subopt_price ^
        --maze_task -1 ^
        --epochs 5 ^
        --use_price true ^
        --demo_dir demos/corrupted --demo_suffix %%S ^
        --icl_config.log_path %PRICE_CON_ROOT%/!TAG!
)
echo PRICE-conservative sweep done.

REM ----------------------------------------------------------
REM  Step 4c: PRICE-forward (aggressive settings)
REM ----------------------------------------------------------
echo.
echo === Step 4c: PRICE-forward (clean + corrupted) ===
set PRICE_FWD_ROOT=experiments\breaking\results\price_forward_suboptimal
for %%C in (clean vr5 vr10 vr20 vr50) do mkdir "!PRICE_FWD_ROOT!\%%C" 2>nul

python script/planner_icl.py ^
    --icl_config.task AntMaze_UMazeDense-v3 ^
    --icl_config.constraint_type Maze ^
    --icl_config.seed 100 ^
    --icl_config.suffix breaking_subopt_price_forward ^
    --exp_name breaking_subopt_price_forward ^
    --maze_task -1 ^
    --epochs 5 ^
    --use_price true ^
    --icl_config.expert_batch_fraction 0.25 ^
    --icl_config.price.route_all_dispreferred true ^
    --icl_config.price.routed_batch_fraction 1.0 ^
    --icl_config.price.pairs_per_constraint_phase 256 ^
    --icl_config.price.routed_positions_cap 16384 ^
    --icl_config.log_path %PRICE_FWD_ROOT%/clean

for %%S in (_vr5 _vr10 _vr20 _vr50) do (
    set "TAG=%%S"
    set "TAG=!TAG:~1!"
    echo --- PRICE-forward condition: !TAG! ---
    python script/planner_icl.py ^
        --icl_config.task AntMaze_UMazeDense-v3 ^
        --icl_config.constraint_type Maze ^
        --icl_config.seed 100 ^
        --icl_config.suffix breaking_subopt_price_forward ^
        --exp_name breaking_subopt_price_forward ^
        --maze_task -1 ^
        --epochs 5 ^
        --use_price true ^
        --icl_config.expert_batch_fraction 0.25 ^
        --icl_config.price.route_all_dispreferred true ^
        --icl_config.price.routed_batch_fraction 1.0 ^
        --icl_config.price.pairs_per_constraint_phase 256 ^
        --icl_config.price.routed_positions_cap 16384 ^
        --demo_dir demos/corrupted --demo_suffix %%S ^
        --icl_config.log_path %PRICE_FWD_ROOT%/!TAG!
)
echo PRICE-forward sweep done.

echo.
echo ============================================================
echo  All runs complete. Results under experiments\breaking\results\
echo ============================================================
