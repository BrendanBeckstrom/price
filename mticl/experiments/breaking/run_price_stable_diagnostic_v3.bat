@echo off
REM PRICE stability diagnostic v3: vr20 only, frozen-growing pool, expert_batch_fraction=0.20.
REM Same as v2 but with lower expert batch fraction to test sensitivity.
setlocal enabledelayedexpansion

cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
echo Working directory: %CD%

set "RESULT_ROOT=experiments\breaking\results\price_stable_diagnostic_v3"
set "OUTDIR=%RESULT_ROOT%\violate_20"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "COMMON=--icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.seed 100 --icl_config.suffix breaking_price_stable_diag_v3 --exp_name breaking_price_stable_diag_v3 --maze_task -1 --epochs 5 --use_price true --icl_config.expert_batch_fraction 0.20 --icl_config.constraint_steps 2500 --icl_config.price.route_all_dispreferred true --icl_config.price.routed_batch_fraction 1.0 --icl_config.price.pairs_per_constraint_phase 256 --icl_config.price.routed_positions_cap 16384 --demo_dir demos/corrupted --demo_suffix _vr20 --icl_config.log_path %OUTDIR%"

echo.
echo =================================================================
echo   PRICE Stability Diagnostic v3 -- vr20 (expert_batch_fraction=0.20)
echo =================================================================

for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "START_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
python script\planner_icl.py %COMMON% > "%OUTDIR%\run.log" 2>&1
type "%OUTDIR%\run.log"
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "END_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
set /a "ELAPSED=END_S - START_S"
if !ELAPSED! lss 0 set /a "ELAPSED=ELAPSED + 86400"
echo !ELAPSED! > "%OUTDIR%\time.txt"
echo ===== vr20 finished in !ELAPSED!s =====

REM --- Post-run diagnostics ---
echo.
echo =================================================================
echo   Diagnostic v3 Summary
echo =================================================================

REM Final IoU
set "IOU=N/A"
for /f "delims=" %%I in ('python -c "import sys; sys.path.insert(0,'.'); from experiments.breaking.compare_iou_utils import iou_for_run_dir; v=iou_for_run_dir('!OUTDIR:\=/!'); print(f'{v:.4f}' if v is not None else 'N/A')" 2^>nul') do set "IOU=%%I"
echo Final IoU: !IOU!

REM Exhaustion count
set "EXH=0"
if exist "%OUTDIR%\run.log" (
    for /f %%N in ('findstr /c:"WARNING: gen_valid_demo failed" "%OUTDIR%\run.log" ^| find /c /v ""') do set "EXH=%%N"
)
echo gen_valid_demo exhaustions: !EXH!

REM Wall violation counts per epoch
echo.
echo Wall violations per epoch:
findstr /c:"epoch trajectories have wall_violations" "%OUTDIR%\run.log" 2>nul || echo (not found in log)

REM Routed pool size per epoch
echo.
echo Routed pool size per epoch:
findstr /c:"routed pool size" "%OUTDIR%\run.log" 2>nul || echo (not found in log)

echo.
echo Elapsed time: !ELAPSED!s
echo Results saved under %OUTDIR%
echo =================================================================
