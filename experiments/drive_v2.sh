#!/bin/zsh
# Detached v2 driver: runs remaining phases in 3 parallel workers,
# writes sentinel files on completion. Immune to harness task reaping.
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/bin/python
LOG=results/v2/_driver
mkdir -p "$LOG"

(
  $PY experiments/run_v2.py --phase 2 > "$LOG/w_e2m.log" 2>&1
  touch "$LOG/done_e2m"
) &
(
  $PY experiments/run_v2.py --phase 4 > "$LOG/w_e4.log" 2>&1
  touch "$LOG/done_e4"
) &
(
  $PY experiments/run_v2.py --phase 1 > "$LOG/w_misc.log" 2>&1
  $PY experiments/run_v2.py --phase 5 >> "$LOG/w_misc.log" 2>&1
  $PY experiments/run_v2.py --phase 2 --dataset fashion >> "$LOG/w_misc.log" 2>&1
  touch "$LOG/done_misc"
) &
wait
touch "$LOG/done_all"
