#!/usr/bin/env bash
# Pull small result files of every run from the VM into outputs/ablation/<run>/ (tar on VM, one download).
S=${1:-dv-abl}
cd "$(dirname "$0")/.."
D=outputs/ablation; mkdir -p $D
RSH_WALL=200 RSH_TIMEOUT=180 scripts/rsh.sh $S "cd /content/out && tar czf /content/abl_results.tgz --ignore-failed-read queue.log post.log abl/*/config.json abl/*/train_log.jsonl abl/*/*stdout.log abl/*/eval.json abl/*/rows.json abl/*/eval_c.json abl/*/rows_c.json abl/*/latency.json 2>/dev/null; ls -la /content/abl_results.tgz" >/dev/null 2>&1
TMP=$(mktemp -d)
timeout 600 colab-pool download $S /content/abl_results.tgz $TMP/r.tgz >/dev/null 2>&1 || { echo "[pull] download failed"; exit 1; }
tar xzf $TMP/r.tgz -C $TMP 2>/dev/null
cp -f $TMP/queue.log $TMP/post.log $D/ 2>/dev/null
for d in $TMP/abl/*/; do r=$(basename $d); mkdir -p $D/$r; cp -f $d/* $D/$r/; done
rm -rf $TMP
echo "[pull] ok $(date +%H:%M) $(ls $D | tr '\n' ' ')"
