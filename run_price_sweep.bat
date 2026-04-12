@echo off
REM ============================================================
REM  PRICE sweep only (skips regression + smoke test)
REM  Run from Anaconda prompt in the repo root:
REM    run_price_sweep.bat
REM ============================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"
cd mticl

REM ----------------------------------------------------------
REM  PRICE-conservative (clean + corrupted)
REM ----------------------------------------------------------
echo === PRICE-conservative (clean + corrupted) ===
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
REM  PRICE-forward (aggressive settings)
REM ----------------------------------------------------------
echo.
echo === PRICE-forward (clean + corrupted) ===
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
echo  All PRICE runs complete. Results under experiments\breaking\results\
echo ============================================================
