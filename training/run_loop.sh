#!/usr/bin/env bash
# Auto-restart wrapper for train_last2.py on a flaky GPU (Ubuntu box).
#
# The GPU on this box throws "CUDA error: unspecified launch failure" after a roughly
# CONSTANT runtime under load -- a thermal/power/driver instability, not a script bug.
# This wrapper relaunches a FRESH process on every non-zero exit; train_last2.py resumes
# each phase from last.pt, skips finished sizes, and uploads a model to W&B after the first
# epoch of every run. So the job crawls forward and keeps banking checkpoints to the cloud
# despite the crashes, with no manual restarts.
#
# Usage:
#   bash training/run_loop.sh                       # all sizes, defaults
#   bash training/run_loop.sh --sizes m --batch 8   # args pass through to train_last2.py
#
# Recommended on this box (reduce crash frequency -- see notes at bottom):
#   sudo nvidia-smi -pl 300        # cap power -> less heat/less transient spikes
#   bash training/run_loop.sh --batch 8
#
# Stop it: Ctrl+C (the trap below stops the loop), then:  pkill -f train_last2.py
#
# NOTE: this recovers from CRASHES (process exits). If it truly HANGS (no log output for a
# long time, process alive), Ctrl+C the wrapper and run `pkill -f train_last2.py`; the next
# launch resumes from last.pt.
set -u
trap 'echo "[run_loop] interrupted -- stopping."; exit 130' INT TERM

cd "$(dirname "$0")/.."            # repo root
MAX="${RUN_LOOP_MAX:-1000}"        # safety cap on restarts
SLEEP="${RUN_LOOP_SLEEP:-30}"      # pause between restarts (also lets the GPU cool)
n=0
while :; do
  n=$((n + 1))
  echo "[run_loop] attempt $n/$MAX : $(date) : python training/train_last2.py $*"
  python training/train_last2.py "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "[run_loop] train_last2.py finished cleanly (exit 0). All done."
    break
  fi
  if [ "$code" -eq 3 ]; then
    echo "[run_loop] exit 3 = fatal config (W&B not logged in / offline mode). NOT restarting."
    echo "[run_loop] Fix it (e.g. 'wandb login') and relaunch. Restarting can't fix a config error."
    break
  fi
  echo "[run_loop] exit code $code -- crash/GPU-fault recovery; restarting in ${SLEEP}s. (Ctrl+C to stop)"
  if [ "$n" -ge "$MAX" ]; then
    echo "[run_loop] hit MAX=$MAX restarts; stopping. Investigate GPU stability (see header notes)."
    break
  fi
  sleep "$SLEEP"
done
