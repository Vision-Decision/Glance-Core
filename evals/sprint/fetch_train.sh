#!/usr/bin/env bash
# GUI sprint training data on the VM (after setup_vm.sh). Raw data stays on the VM; nothing comes to the laptop.
#   OS-Atlas desktop (linux, macOS) + mobile (AndroidWorld, UIBert), AGUVIS web (guienv), guisyn-v0 (private, verified
#   negatives), GroundCUA (MIT; N screenshots per app, rate-limited fetch). Then evals/sprint/prep_train.py.
set -uo pipefail
R=/content/data; mkdir -p $R/gui
export HF_HUB_DISABLE_PROGRESS_BARS=1 PIP_DISABLE_PIP_VERSION_CHECK=1
PER_APP=${PER_APP:-250}
pip install -q imagehash 2>/dev/null
echo "[ft] $(date -u +%T) start"; df -h /content | tail -1
# GroundCUA in the background (slowest: per-file downloads with 429 backoff)
(timeout 14400 python /content/dv/scripts/v1/fetch_groundcua.py --per_app $PER_APP --out $R/gui/groundcua --workers 6 > /content/groundcua.log 2>&1; echo "[ft] groundcua rc=$?" >> /content/groundcua.log) &
python - <<'PY'
from huggingface_hub import snapshot_download
R = "/content/data"
jobs = [
  ("Vision-Decision/guisyn-v0", f"{R}/guisyn_raw", None),
  ("OS-Copilot/OS-Atlas-data", f"{R}/gui/osatlas", ["desktop_domain/linux_*", "desktop_domain/macos_*", "mobile_domain/UIBert.zip",
       "mobile_domain/uibert_raw.json", "mobile_domain/mobile_images.zip", "mobile_domain/widget_captioning.json",
       "mobile_domain/ricosca.json", "mobile_domain/aw_mobile.json"]),
  ("xlangai/aguvis-stage1", f"{R}/gui/aguvis", ["guienv.json", "guienvs.zip"]),
]
for rid, d, pat in jobs:
    try:
        snapshot_download(rid, repo_type="dataset", local_dir=d, allow_patterns=pat, max_workers=16)
        print("[ft] ok", rid, flush=True)
    except Exception as e:
        print("[ft] FAIL", rid, repr(e)[:300], flush=True)
PY
cd $R/gui/osatlas
for z in desktop_domain/*.zip mobile_domain/*.zip; do
  [ -f $z.done ] && continue
  d=$(dirname $z)/img; mkdir -p $d; unzip -q -o $z -d $d && touch $z.done && rm -f $z; echo "[ft] unzipped $z"
done
cd $R/gui/aguvis
[ -f guienvs.zip.done ] || { unzip -q -o guienvs.zip -d img && touch guienvs.zip.done && rm -f guienvs.zip; echo "[ft] unzipped guienvs"; }
cd /content/dv
python scripts/guisyn/to_v1.py --shards $R/guisyn_raw --data_root $R --out $R/prep_gui --extract --max_vis 2048 && echo "[ft] guisyn ok"
echo "[ft] $(date -u +%T) waiting for groundcua"; wait
tail -2 /content/groundcua.log
du -sh $R/gui/* $R/guisyn 2>/dev/null; df -h /content | tail -1
echo "[ft] $(date -u +%T) DONE"
