#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

wait_for_file() {
  local target="$1"
  while [[ ! -f "$target" ]]; do
    sleep 60
  done
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Waiting for current FixMatch+IFCF ratio30 run to finish"
wait_for_file "results/ALL_DONE_FIXMATCH_IFCF_BUS_LD05_RATIO30"

for ratio in 5 10 15 30; do
  log "Starting supervised ratio ${ratio}"
  RATIO="$ratio" bash run_supervised_bus_ld05_ratio.sh
done

log "Starting FixMatch ratio15"
RATIO=15 bash run_fixmatch_bus_ld05_ratio.sh

log "Starting FixMatch+IFCF ratio15"
RATIO=15 bash run_fixmatch_ifcf_bus_ld05_ratio.sh

log "Starting FixMatch+IFCF ratio10 extra seeds 6-10"
RATIO=10 SEEDS="6 7 8 9 10" RESET_SUMMARY=0 bash run_fixmatch_ifcf_bus_ld05_ratio.sh

touch "results/ALL_DONE_BUS_RATIO_LD05_FOLLOWUPS"
log "All queued BUS ratio follow-up jobs finished"
