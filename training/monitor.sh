#!/usr/bin/env bash
# Log GPU + system-RAM stats so the NEXT crash can be diagnosed after the fact.
# Run it alongside training, ideally in its own tmux window:
#     bash training/monitor.sh
# After a crash, look at the tail of the log to see what spiked right before it died:
#     tail -n 20 ~/njord_monitor.log
#   - gpu_temp near ~84 C, or power at the limit  -> thermal/power  (lower `nvidia-smi -pl`)
#   - ram_used ~= ram_total                       -> system OOM     (lower --workers)
#   - nothing obvious                             -> likely the GPU driver crashing X;
#                                                    check dmesg/journalctl (see bottom).
set -u
LOG="${1:-$HOME/njord_monitor.log}"
trap 'echo "[monitor] stopped."; exit 0' INT TERM
echo "[monitor] logging -> $LOG  (Ctrl+C to stop)"
if [ ! -s "$LOG" ]; then
  echo "timestamp,gpu_temp_C,gpu_util_%,power_W,power_limit_W,vram_used_MiB,vram_total_MiB,ram_used_MiB,ram_total_MiB" >> "$LOG"
fi
while :; do
  ts=$(date -Iseconds)
  g=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$g" ] && g="NA,NA,NA,NA,NA,NA"
  r=$(free -m | awk '/Mem:/{print $3","$2}')
  echo "$ts,$g,$r" >> "$LOG"
  sleep 5
done
