@echo off
REM PRICE Stable Diagnostic v2 -- Final Run: all five conditions (clean, vr5, vr10, vr20, vr50).
REM Uses the best-performing v2 hyperparameters: frozen-growing routed pool,
REM routed_batch_fraction=1.0, constraint_steps=2500, expert_batch_fraction=0.25.
setlocal enabledelayedexpansion

cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
echo Working directory: %CD%

set "RESULT_ROOT=experiments\breaking\results\final_results"
set "COMMON=--icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --icl_config.seed 100 --icl_config.suffix final_price --exp_name final_price --maze_task -1 --epochs 5 --use_price true --icl_config.expert_batch_fraction 0.25 --icl_config.constraint_steps 2500 --icl_config.price.route_all_dispreferred true --icl_config.price.routed_batch_fraction 1.0 --icl_config.price.pairs_per_constraint_phase 256 --icl_config.price.routed_positions_cap 16384"

REM === clean ===
set "COND=clean"
set "OUTDIR=%RESULT_ROOT%\%COND%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo.
echo ===== Running condition: %COND% =====
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "START_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
python script\planner_icl.py %COMMON% --icl_config.log_path "%OUTDIR%" > "%OUTDIR%\run.log" 2>&1
type "%OUTDIR%\run.log"
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "END_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
set /a "ELAPSED_clean=END_S - START_S"
if !ELAPSED_clean! lss 0 set /a "ELAPSED_clean=ELAPSED_clean + 86400"
echo !ELAPSED_clean! > "%OUTDIR%\time.txt"
echo ===== %COND% finished in !ELAPSED_clean!s =====

REM === vr5 ===
set "COND=vr5"
set "OUTDIR=%RESULT_ROOT%\%COND%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo.
echo ===== Running condition: %COND% =====
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "START_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
python script\planner_icl.py %COMMON% --demo_dir demos/corrupted --demo_suffix _vr5 --icl_config.log_path "%OUTDIR%" > "%OUTDIR%\run.log" 2>&1
type "%OUTDIR%\run.log"
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "END_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
set /a "ELAPSED_vr5=END_S - START_S"
if !ELAPSED_vr5! lss 0 set /a "ELAPSED_vr5=ELAPSED_vr5 + 86400"
echo !ELAPSED_vr5! > "%OUTDIR%\time.txt"
echo ===== %COND% finished in !ELAPSED_vr5!s =====

REM === vr10 ===
set "COND=vr10"
set "OUTDIR=%RESULT_ROOT%\%COND%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo.
echo ===== Running condition: %COND% =====
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "START_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
python script\planner_icl.py %COMMON% --demo_dir demos/corrupted --demo_suffix _vr10 --icl_config.log_path "%OUTDIR%" > "%OUTDIR%\run.log" 2>&1
type "%OUTDIR%\run.log"
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "END_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
set /a "ELAPSED_vr10=END_S - START_S"
if !ELAPSED_vr10! lss 0 set /a "ELAPSED_vr10=ELAPSED_vr10 + 86400"
echo !ELAPSED_vr10! > "%OUTDIR%\time.txt"
echo ===== %COND% finished in !ELAPSED_vr10!s =====

REM === vr20 ===
set "COND=vr20"
set "OUTDIR=%RESULT_ROOT%\%COND%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo.
echo ===== Running condition: %COND% =====
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "START_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
python script\planner_icl.py %COMMON% --demo_dir demos/corrupted --demo_suffix _vr20 --icl_config.log_path "%OUTDIR%" > "%OUTDIR%\run.log" 2>&1
type "%OUTDIR%\run.log"
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "END_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
set /a "ELAPSED_vr20=END_S - START_S"
if !ELAPSED_vr20! lss 0 set /a "ELAPSED_vr20=ELAPSED_vr20 + 86400"
echo !ELAPSED_vr20! > "%OUTDIR%\time.txt"
echo ===== %COND% finished in !ELAPSED_vr20!s =====

REM === vr50 ===
set "COND=vr50"
set "OUTDIR=%RESULT_ROOT%\%COND%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
echo.
echo ===== Running condition: %COND% =====
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "START_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
python script\planner_icl.py %COMMON% --demo_dir demos/corrupted --demo_suffix _vr50 --icl_config.log_path "%OUTDIR%" > "%OUTDIR%\run.log" 2>&1
type "%OUTDIR%\run.log"
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a "END_S=(1%%a-100)*3600 + (1%%b-100)*60 + (1%%c-100)"
set /a "ELAPSED_vr50=END_S - START_S"
if !ELAPSED_vr50! lss 0 set /a "ELAPSED_vr50=ELAPSED_vr50 + 86400"
echo !ELAPSED_vr50! > "%OUTDIR%\time.txt"
echo ===== %COND% finished in !ELAPSED_vr50!s =====

REM --- Summary table ---
echo.
echo =================================================================
echo   PRICE Stable Diagnostic v2 -- Final Results Summary
echo =================================================================
echo Condition    Final IoU    Exhaustions    Time (s)
echo -----------------------------------------------------------------

for %%C in (clean vr5 vr10 vr20 vr50) do (
    set "OUTDIR=%RESULT_ROOT%\%%C"
    set "IOU=N/A"
    set "EXH=0"

    for /f "delims=" %%I in ('python -c "import sys; sys.path.insert(0,'.'); from experiments.breaking.compare_iou_utils import iou_for_run_dir; v=iou_for_run_dir('!OUTDIR:\=/!'); print(f'{v:.4f}' if v is not None else 'N/A')" 2^>nul') do set "IOU=%%I"

    if exist "!OUTDIR!\run.log" (
        for /f %%N in ('findstr /c:"WARNING: gen_valid_demo failed" "!OUTDIR!\run.log" ^| find /c /v ""') do set "EXH=%%N"
    )

    set "ELAPSED=?"
    if exist "!OUTDIR!\time.txt" set /p ELAPSED=<"!OUTDIR!\time.txt"

    echo %%C           !IOU!        !EXH!            !ELAPSED!
)

echo =================================================================
echo Results saved under %RESULT_ROOT%
