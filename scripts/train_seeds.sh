#!/usr/bin/env bash
# Four more seeds for each configuration (seed 42 is the released run); about 5 minutes each
# on an Apple M5 laptop. Then: python scripts/compare_seeds.py
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/seeds
for s in 1 2 3 4; do
  for cfg in meta nometa; do
    out="runs/seeds/${cfg}_s${s}"
    [ -f "$out/metrics.json" ] && continue
    extra=""; [ "$cfg" = nometa ] && extra="--no-meta"
    echo "=== seed $s $cfg $(date +%H:%M:%S)"
    .venv/bin/python train_ptbxl.py --data data/ptb-xl --epochs 30 --seed "$s" --out "$out" $extra > "$out.log" 2>&1
    tail -n 3 "$out.log" | grep -E "test_macro_auc|Error" || true
  done
done
echo "=== all done $(date +%H:%M:%S)"
