#!/usr/bin/env bash
# Decision Index public runs for Glance v1 on one Colab G4 (RTX PRO 6000). Run ON THE VM, detached (see di/RUN.md).
#   bash /content/dv/di/colab_run.sh setup|build_text|build_vision|probe|agree|text|score_text|vision|all
# Nothing large comes from the laptop: only /content/dv.tgz (code: dv/, di/, eval.py, configs/) and /content/kit.tgz
# (decision-index @25c6ec7). Weights come from the public HF release Vision-Decision/Glance-4B-preview (pinned revision);
# both suites are rebuilt here from their pinned upstream sources (they may not be mirrored: see RUN.md).
# Needs an HF token on the VM (HLE and Winoground are gated; Winoground's terms must be accepted by the user first).
# Every stage resumes: re-running keeps finished downloads / result rows (errors are retried).
set -euo pipefail
STAGE=${1:-all}
OUT=/content/out/di
WORK=/content/di/work           # text rebuild workspace (~17 GB)
SUITE=/content/di/suite-0.3
VWORK=/content/di/vwork         # vision rebuild -> $VWORK/vision
VIS=$VWORK/vision
FONT=/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf
# The rebuilt selected-rows may differ from the lab's through SGD only (RUN.md "Suite integrity"); strict verification
# would refuse them. NOVERIFY=1 (default) passes --no-verify; set NOVERIFY=0 once `suite verify` matches.
NV=(); [ "${NOVERIFY:-1}" = 1 ] && NV=(--no-verify)
# Weights: default = release repo pinned in di/engine.py. Later versions: MODEL=<private repo> REVISION=<sha> (HF token).
OPTS=(--option "model=${MODEL:-Vision-Decision/Glance-4B-preview}")
[ -n "${REVISION:-}" ] && OPTS+=(--option "revision=$REVISION")
export PYTHONPATH=/content/dv HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p $OUT /content/di
cd /content/dv

setup() {
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  pip uninstall -y -q torchao || true
  pip install -q "transformers==5.23.0" "flash-linear-attention==0.6.0" "peft>=0.17" accelerate safetensors pyarrow pandas httpx
  if [ ! -d /content/decision-index ]; then
    if [ -f /content/kit.tgz ]; then mkdir -p /content/decision-index && tar xzf /content/kit.tgz -C /content/decision-index;
    else git clone -q https://github.com/apolinario/decision-index /content/decision-index && git -C /content/decision-index checkout -q 25c6ec7; fi
  fi
  # Pillow 12.3.0: the kit's pinned image digests (InfographicVQA, CORD resizes) reproduce with 9.5.0 / 12.3.0, not 13
  pip install -q -e "/content/decision-index[rebuild,vision]" "pillow==12.3.0"
  python -c "from huggingface_hub import whoami; print('hf user:', whoami()['name'])"   # fails early if no token
  python - <<'EOF'
from huggingface_hub import snapshot_download as s, hf_hub_download as h
print(s('Qwen/Qwen3.5-4B', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja']))
import os, re
src = open('/content/dv/di/engine.py').read()
repo = os.environ.get('MODEL') or 'Vision-Decision/Glance-4B-preview'
rev = os.environ.get('REVISION') or re.search(r'revision="([0-9a-f]{40})"', src).group(1)
for f in ('glance_config.json', 'adapter_model.safetensors', 'heads.safetensors'):
    print(h(repo, f, revision=rev))
EOF
  python -c "import torch, fla, transformers, PIL; print(torch.__version__, torch.version.cuda, fla.__version__, transformers.__version__, PIL.__version__)"
}

build_text() {   # ~7 GB of pinned downloads, ~17 GB working space, ~40-60 min
  [ -f $SUITE/manifest.json ] && { echo "text suite already imported"; return; }
  local R=$WORK/artifacts/benchmark-suite/raw
  # HoVer's 2.2 GB DB is the flakiest download: fetch it resumably first (the kit reuses a file with the pinned sha256)
  mkdir -p $R/hover
  if ! echo "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f  $R/hover/wiki_wo_links.db" | sha256sum -c --status 2>/dev/null; then
    rm -f $R/hover/wiki_wo_links.db
    # one resumable attempt per loop turn (curl --retry together with -C - can re-append bytes after a dropped connection)
    for i in $(seq 1 30); do
      [ "$(stat -c %s $R/hover/wiki_wo_links.db.part 2>/dev/null || echo 0)" -ge 2156273664 ] && break
      curl -sSL -C - -o $R/hover/wiki_wo_links.db.part https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db || sleep 10
    done
    [ "$(stat -c %s $R/hover/wiki_wo_links.db.part)" -eq 2156273664 ] || { echo "HoVer size wrong"; rm -f $R/hover/wiki_wo_links.db.part; return 1; }
    mv $R/hover/wiki_wo_links.db.part $R/hover/wiki_wo_links.db
    echo "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f  $R/hover/wiki_wo_links.db" | sha256sum -c
  fi
  for i in 1 2 3 4; do
    python -m decision_index suite rebuild --work $WORK > $OUT/rebuild_text_$i.log 2>&1 && break
    tail -5 $OUT/rebuild_text_$i.log
    for d in $R/repos/*; do git -C "$d" rev-parse HEAD >/dev/null 2>&1 || { echo "removing broken clone $d"; rm -rf "$d"; }; done
    sleep 30
  done
  grep -h '"byte_identical"\|release_v2_skipped' $OUT/rebuild_text_*.log | tail -3 || true
  # the kit skips the v2 cut + GSM8K v3 when release-v1 rows are not byte-identical; finish and import either way
  python -m di.finish_suite --work $WORK --suite $SUITE
  python -m decision_index suite verify --dir $SUITE | grep -E '"(match|uncompressed_match|added_match|gsm8k_match)"' || true
}

build_vision() {  # ~5.7 GB of pinned downloads, ~4.4 GB on disk, ~20-30 min
  [ -f $VIS/vision-rows.jsonl ] && { echo "vision suite already built"; return; }
  if [ ! -f $FONT ]; then   # Mind2Web box labels are drawn in Arial Bold (the kit's pins were made with macOS Arial Bold)
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q ttf-mscorefonts-installer > /dev/null
  fi
  python -m decision_index suite rebuild --vision --work $VWORK --font $FONT \
    --sets full_cvbench full_blink full_realworldqa full_charxiv full_infovqa full_mind2web full_cord full_funsd \
           full_hateful_memes full_rbench_m full_mmmu_pro_vision > $OUT/rebuild_vision.log 2>&1 || true
  # Winoground is gated: it builds only if the token's account accepted its terms (manual step for the user)
  if python -m decision_index suite rebuild --vision --work $VWORK --font $FONT --sets full_winoground > $OUT/rebuild_winoground.log 2>&1; then
    echo "winoground built"
  elif [ "${ALLOW_NO_WINOGROUND:-0}" = 1 ]; then
    echo "WINOGROUND NOT BUILT (gated); ALLOW_NO_WINOGROUND=1: writing rows from the other sets (run is NOT complete)"
  else
    echo "WINOGROUND NOT BUILT (gated: accept the terms at hf.co/datasets/facebook/winoground, then re-run build_vision)"
    rm -f $VIS/vision-rows.jsonl; return 1
  fi
  python - <<EOF
from pathlib import Path
from decision_index.vision import spec, rows as R
root = Path("$VIS"); out = []
for s in spec.SETS:   # identical to rows.write_rows when every set is present; skips a missing (gated) set
    if (root / "full" / s / "items.jsonl").exists():
        out += [R.to_row(root, it) for it in R.select(s, R.load_items(root, s))]
    else:
        print("missing set", s)
with open(root / R.ROWS_FILE, "w", encoding="utf-8") as f:
    f.writelines(R.dumps(r) + "\n" for r in out)
print("vision rows", len(out))
EOF
  python -m decision_index vision verify --dir $VIS > $OUT/vision_verify.json
  echo "vision verify false flags: $(grep -c '": false' $OUT/vision_verify.json || true)  (Mind2Web flags = font differs from the lab's)"
}

probe() {   # latency-v1-style pre-flight (750 + 10 rows); writes latency_probe.json
  export HF_HUB_OFFLINE=1
  python -m di.latency_probe --suite $SUITE "${OPTS[@]}" --out $OUT/latency_probe.json
}

agree() {   # determinism: the same 3,000-row stratified sample in two fresh processes
  export HF_HUB_OFFLINE=1
  [ -f $OUT/agree_rows.jsonl.gz ] || python -m decision_index suite sample --dir $SUITE --n 3000 --out $OUT/agree_rows.jsonl.gz
  for r in a b; do
    python -m decision_index run --engine di.engine:GlanceEngine "${OPTS[@]}" --rows $OUT/agree_rows.jsonl.gz --out $OUT/agree_$r --fresh
  done
  python -m di.agreement $OUT/agree_a/results.jsonl $OUT/agree_b/results.jsonl | tee $OUT/agreement.json
}

text() {    # full public 0.3 text suite (140,178 scoreable requests), scored at the end
  export HF_HUB_OFFLINE=1
  if [ -f /content/di/shard.conf ]; then   # "K CUTS", e.g. "0 0,11,20": this VM runs shard K only (di/shard.py)
    read K CUTS < /content/di/shard.conf
    [ -f /content/di/shard$K.jsonl.gz ] || python -m di.shard make --suite $SUITE --k $K --cuts $CUTS --out /content/di/shard$K.jsonl.gz
    python -m decision_index run --rows /content/di/shard$K.jsonl.gz --engine di.engine:GlanceEngine "${OPTS[@]}" --out $OUT/text-shard$K
  else
    python -m decision_index pipeline --suite-dir $SUITE "${NV[@]}" --engine di.engine:GlanceEngine "${OPTS[@]}" --out $OUT/text-glance-v1
  fi
}

score_text() {   # after all shards: merge their results.jsonl (payload stripped) and score with the kit
  mkdir -p $OUT/text-glance-v1
  python -m di.shard merge $OUT/text-shard*/results.jsonl /content/di/shard_results/*.jsonl.gz --out $OUT/text-glance-v1/results.jsonl
  python -m decision_index score --suite-dir $SUITE --results $OUT/text-glance-v1/results.jsonl --engine di.engine:GlanceEngine
}

vision() {  # full public vision suite (12,210 rows); ONE worker so every row is timed alone
  export HF_HUB_OFFLINE=1
  VNV=(); { [ "${ALLOW_NO_WINOGROUND:-0}" = 1 ] || [ "${NOVERIFY_VISION:-0}" = 1 ]; } && VNV=(--no-verify)
  python -m decision_index vision pipeline --dir $VIS "${VNV[@]}" --engine di.engine:GlanceEngine "${OPTS[@]}" --option max_vis_tokens=1600 \
    --workers 1 --out $OUT/vision-glance-v1
}

case $STAGE in
  setup) setup ;; build_text) build_text ;; build_vision) build_vision ;;
  probe) probe ;; agree) agree ;; text) text ;; score_text) score_text ;; vision) vision ;;
  all) setup; build_text; probe; agree; text; build_vision; vision ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "[di] $STAGE DONE $(date)"
