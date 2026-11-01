#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CAUSAL_MODEL_CANDIDATE=causal_fullband_spatial_tcn_v1
exec bash run_light_causal.sh "$@"
