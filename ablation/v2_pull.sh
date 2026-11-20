#!/usr/bin/env bash
# Pull small result files from a VM into outputs/ablation_v2/runs/<run>/ (and logs into outputs/ablation_v2/logs/<S>/).
# ablation/v2_pull.sh SESSION
S=${1:?session}
cd "$(dirname "$0")/.."
D=outputs/ablation_v2; mkdir -p $D/runs $D/logs/$S
RSH_WALL=240 RSH_TIMEOUT=200 scripts/rsh.sh $S "cd /content/out && tar czf /content/v2_results.tgz --ignore-failed-read *.log abl/*/config.json abl/*/train_log.jsonl abl/*/*stdout.log abl/*/eval*.json abl/*/rows*.json abl/*/latency*.json v2/ 2>/dev/null; ls -la /content/v2_results.tgz" >/dev/null 2>&1
TMP=$(mktemp -d)
timeout 900 colab-pool download $S /content/v2_results.tgz $TMP/r.tgz >/dev/null 2>&1 || { echo "[pull] $S download failed"; rm -rf $TMP; exit 1; }
tar xzf $TMP/r.tgz -C $TMP 2>/dev/null
cp -f $TMP/*.log $D/logs/$S/ 2>/dev/null
for d in $TMP/abl/*/; do r=$(basename $d); mkdir -p $D/runs/$r; cp -f $d/* $D/runs/$r/ 2>/dev/null; done
[ -d $TMP/v2 ] && { mkdir -p $D/raw; cp -rf $TMP/v2/* $D/raw/; }
rm -rf $TMP
echo "[pull] $S ok $(date +%H:%M)"
