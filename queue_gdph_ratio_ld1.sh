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

log "Waiting for BUS tail queue to finish"
wait_for_file "results/ALL_DONE_SUPERVISED_BUS_LD05_ALL"

log "Starting GDPH fixmatch+if ratio 10"
METHOD=fixmatch_if RATIO=10 bash run_gdph_ratio_ld1.sh

log "Starting GDPH supervised ratio 15"
METHOD=supervised RATIO=15 bash run_gdph_ratio_ld1.sh

log "Starting GDPH fixmatch ratio 15"
METHOD=fixmatch RATIO=15 bash run_gdph_ratio_ld1.sh

log "Starting GDPH fixmatch+if ratio 15"
METHOD=fixmatch_if RATIO=15 bash run_gdph_ratio_ld1.sh

log "Starting GDPH fixmatch+if ratio 30"
METHOD=fixmatch_if RATIO=30 bash run_gdph_ratio_ld1.sh

touch "results/ALL_DONE_GDPH_RATIO_LD1"
log "All queued GDPH ratio ld1 jobs finished"
