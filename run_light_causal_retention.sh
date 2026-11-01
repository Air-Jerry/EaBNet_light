#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CAUSAL_FREQUENCY_ATTENTION_ENTRY=train_light_causal_retention.py
exec bash run_light_causal_frequency_attention.sh "$@"
