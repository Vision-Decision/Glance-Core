#!/usr/bin/env bash
# v1.1 extra downloads (run after scripts/v1/fetch.sh). Idempotent-ish; logs and continues on failure.
R=/content/data; mkdir -p $R/gui/osatlas/desktop_domain/img_win $R/gui/groundcua $R/gui/jedi
export HF_HUB_DISABLE_PROGRESS_BARS=1
echo "[f11] $(date) start"; df -h /content | tail -1
apt-get install -y -qq libarchive-tools >/dev/null 2>&1; which bsdtar
# 1) OS-Atlas Windows desktop (4 split parts of one zip, ~71 GB): stream-extract, never stored
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("OS-Copilot/OS-Atlas-data", "desktop_domain/windows_splited.json", repo_type="dataset", local_dir="/content/data/gui/osatlas")
PY
if [ ! -f $R/gui/osatlas/desktop_domain/img_win/.done ]; then
  B=https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data/resolve/main/desktop_domain
  (for p in aa ab ac ad; do curl -sL --retry 20 --retry-delay 30 $B/windows_image_$p; done) | bsdtar -xf - -C $R/gui/osatlas/desktop_domain/img_win \
    && touch $R/gui/osatlas/desktop_domain/img_win/.done; echo "[f11] windows extract rc=$? files=$(find $R/gui/osatlas/desktop_domain/img_win -type f | wc -l)"
fi
# 2) Jedi: os_layout + component_v1_130k grounding (Apache-2.0)
python - <<'PY'
from huggingface_hub import snapshot_download
for pat in [["datasets/layout/os_layout_v1_grounding/*", "images/os_layout_data/*"],
            ["datasets/component/generated/component_v1_130k/*", "images/component_v1_130k/*"]]:
    try:
        snapshot_download("xlangai/Jedi", repo_type="dataset", local_dir="/content/data/gui/jedi", allow_patterns=pat, max_workers=4)
        print("[f11] jedi ok", pat, flush=True)
    except Exception as e:
        print("[f11] jedi FAIL", pat, repr(e)[:200], flush=True)
PY
cd $R/gui/jedi/images && for d in os_layout_data component_v1_130k; do [ -f $d/.done ] || { cat $d/$d.tar.gz* | tar xzf - -C $d && touch $d/.done && rm -f $d/$d.tar.gz*; echo "[f11] untar $d"; }; done
# 3) GroundCUA: per-file with backoff, low concurrency (json + png of N screenshots per app)
python /content/dv/scripts/v1/fetch_groundcua.py --per_app 80 --out $R/gui/groundcua
echo "[f11] $(date) DONE"; df -h /content | tail -1
