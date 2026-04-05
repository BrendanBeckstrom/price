@echo off
setlocal EnableExtensions
REM Run full task_k / K-goals maze planner ICL matrix (Windows).
REM Usage: double-click or from Anaconda Prompt:
REM   cd path\to\price\mticl\experiments\breaking
REM   run_task_k.bat
REM Or: run_task_k.bat --skip-plot   (training only)

cd /d "%~dp0..\.."
if not exist "script\planner_icl.py" (
  echo ERROR: Must run from mticl repo: expected script\planner_icl.py under %CD%
  exit /b 1
)

set PYTHONPATH=%CD%
echo PYTHONPATH=%PYTHONPATH%
echo.

if /i "%~1"=="--skip-plot" (
  set SKIP_PLOT=1
) else (
  set SKIP_PLOT=0
)

set COMMON=--icl_config.task AntMaze_UMazeDense-v3 --icl_config.constraint_type Maze --epochs 5
set OUT=experiments\breaking\results\task_k

call :run_one k1_goal_0 0 "[0]" k1_goal_0
if errorlevel 1 exit /b 1
call :run_one k1_goal_3 3 "[3]" k1_goal_3
if errorlevel 1 exit /b 1
call :run_one k1_goal_7 7 "[7]" k1_goal_7
if errorlevel 1 exit /b 1
call :run_one k2_goals_0_1 -1 "[0,1]" k2_goals_0_1
if errorlevel 1 exit /b 1
call :run_one k2_goals_0_5 -1 "[0,5]" k2_goals_0_5
if errorlevel 1 exit /b 1
call :run_one k3_goals_0_3_7 -1 "[0,3,7]" k3_goals_0_3_7
if errorlevel 1 exit /b 1
call :run_one k5_goals_0_2_4_6_8 -1 "[0,2,4,6,8]" k5_goals_0_2_4_6_8
if errorlevel 1 exit /b 1
call :run_one k10_baseline -1 "[0,1,2,3,4,5,6,7,8,9]" k10_baseline
if errorlevel 1 exit /b 1

echo.
echo All training runs finished. Outputs under %OUT%
echo.

if "%SKIP_PLOT%"=="1" (
  echo Skipping plot ^(SKIP_PLOT or --skip-plot^).
  exit /b 0
)

if not exist "experiments\breaking\plot_task_k.py" (
  echo WARNING: plot_task_k.py not found; skipping plot.
  exit /b 0
)

echo Running comparison plot...
python experiments\breaking\plot_task_k.py --results-dir "%OUT%" --constraint-epoch 4
if errorlevel 1 (
  echo WARNING: plot_task_k.py failed ^(e.g. missing ground_truth_maze.npy^). Training outputs are still under %OUT%
  exit /b 1
)

echo Done. Default figure: experiments\breaking\task_k_comparison.pdf
goto :eof

:run_one
REM Args: %1=subdir name  %2=maze_task  %3=task_goals  %4=exp_name
set "NAME=%~1"
set "MT=%~2"
set "GOALS=%~3"
set "EXP=%~4"
echo ===== %EXP% =====
mkdir "%OUT%\%NAME%" 2>nul
python script\planner_icl.py %COMMON% --icl_config.log_path "%OUT%\%NAME%" --maze_task %MT% --task_goals "%GOALS%" --exp_name %EXP%
if errorlevel 1 (
  echo FAILED: %EXP%
  exit /b 1
)
exit /b 0
