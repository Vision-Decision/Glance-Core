#!/usr/bin/env bash
# Pull a run's small outputs (eval json, rg/robust rows, logs; no checkpoints) into outputs/sprint_robust/runs/NAME.
# Usage: scripts/sprint_robust/pull.sh SESSION NAME
S=${1:?session}; N=${2:?name}
cd "$(dirname "$0")/../.."
D=outputs/sprint_robust/runs/$N; mkdir -p $D
scripts/rsh.sh "$S" "cd /content/out/sr/$N && tar czf /content/pull_$N.tgz --exclude='*.pt' --exclude='mix.jsonl' --exclude='empty.jsonl' \$(ls -d eval_results.json eval.log mix_stats.json train_log.jsonl config.json rows/rg.jsonl.gz rows/robust_*.jsonl.gz rows/venus.jsonl.gz rows/ss2.jsonl.gz rows/sspro.jsonl.gz rows/grefcoco_val_merged.jsonl.gz v1 base v1gen *.json *.log 2>/dev/null | sort -u)" > /dev/null
colab-pool download "$S" /content/pull_$N.tgz $D/pull.tgz > /dev/null 2>&1 && tar xzf $D/pull.tgz -C $D && rm $D/pull.tgz && ls $D
