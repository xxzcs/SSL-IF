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

log "Waiting for ld0.5 ratio follow-up queue to finish"
wait_for_file "results/ALL_DONE_BUS_RATIO_LD05_FOLLOWUPS"

log "Starting supervised ld1.0 ratio5"
RATIO=5 bash run_supervised_bus_ld10_ratio.sh

log "Starting FixMatch ld1.0 ratio5"
RATIO=5 bash run_fixmatch_bus_ld10_ratio.sh

log "Starting FixMatch+IFCF ld1.0 ratio5"
RATIO=5 bash run_fixmatch_ifcf_bus_ld10_ratio.sh

log "Starting FixMatch+IFCF ld0.5 ratio30 extra seeds 6-10"
RATIO=30 SEEDS="6 7 8 9 10" RESET_SUMMARY=0 bash run_fixmatch_ifcf_bus_ld05_ratio.sh

log "Starting supervised BUS ld0.5 all-labeled"
bash run_supervised_bus_ld05_all.sh

log "Starting queued GDPH ratio ld1 jobs"
bash queue_gdph_ratio_ld1.sh

log "All queued BUS ld1.0 ratio5 jobs finished"
