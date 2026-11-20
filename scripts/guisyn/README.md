# guisyn: synthetic GUI-grounding data (desktop pro apps + verified-absent negatives + web)

Runs on a CPU-only Linux host (built and run on `gen-host`: Ubuntu 24.04, 30 vCPU, no GPU/KVM/Docker). Output goes to the
PRIVATE Hugging Face dataset `Vision-Decision/guisyn-v0` as append-only shards; `to_v1.py` converts it to the v1 GUI pool
format used by `dv/data/prep_v1.py` / `dv/data/mixed.py`.

## Pipelines

| step | file | what |
|---|---|---|
| desktop worker | `desktop_worker.py` (+ `session.py`, `a11y.py`, `filt.py`, `apps.py`) | one Xvfb display + private D-Bus/AT-SPI bus + openbox + one app, run as the unprivileged `guisyn` user; random-but-safe interactions (menus, tabs, toolbar toggles, context menus, hovers, scrolls, app shortcuts, dialogs; an `UNSAFE` name regex blocks quit/save/print/delete/send/...), settle check, AT-SPI tree dump, X stacking, screenshot |
| orchestrator | `run_desktop.py` | keeps N slots busy; resolutions 1920x1080 / 2560x1440 / 2560x1600 / 3840x2160 / 3840x2160 at 2x UI scale |
| web worker | `web_worker.py` | Playwright Chromium on Tranco top sites (list V38GN), viewports 1280-1920 px, DPR 1 and 2, up to 3 scroll positions + at most one same-site link |
| build | `build_shards.py` (+ `exprs.py`) | referring expressions, verified hard negatives, eval-image dedup, shards |
| paraphrase | `paraphrase.py` | optional DeepSeek (Ollama cloud) functional paraphrases from TEXT only (app/role/name/context); run locally, key from `.env` via process env only |
| upload | `cycle.sh` (cron, every 30 min) -> `build_shards.py`, `stats.py`, `upload.py` | incremental upload of complete shards |
| QC | `qc.py`, `draw.py`, `probe.py` | crop sheets for positives, full-frame sheets for negatives, HTML index |
| eval dedup | `evalhash.py` | hashes of ScreenSpot-v2 / ScreenSpot-Pro / VenusBench-GD images fetched with `evals/gui_data.py` |
| samples | `make_samples.py` | synthetic documents the apps open (docx/xlsx/pptx/svg/images/video/audio/pcap/geojson/code) + KiCad demos + FreeCAD test fixtures |
| convert | `to_v1.py` | shards -> `pool_guisyn.jsonl` / `calib_guisyn.jsonl` (v1 GUI record schema) |

## Element filter (desktop)
An element becomes a target only if all of these hold:
- it is SHOWING/VISIBLE and every ancestor is SHOWING (this drops Qt's hidden popup widgets, which report SHOWING at the screen origin);
- its role is a concrete widget or text role; abstract containers are excluded;
- its scale is calibrated (AT-SPI top-level frames vs X client windows), and it is clipped to the screen, its viewport/scroll pane and its top-level, with at least 90% of the box surviving;
- it is at least 6x6 px (times the UI scale) and covers at most 35% of the screen;
- it is not occluded: on a 5x5 grid, at least 22 of 25 points have its own top-level as the topmost X window (override-redirect popups for menu items);
- its pixel content is not flat;
- it passes the AT-SPI hit test: descending `get_accessible_at_point` from the top-level at the box center must land on the element, a descendant or its parent. Hits on another widget, which mean stale extents (LibreOffice's GTK menubar) or a covered widget, are always dropped. GIMP (GTK2), VSCodium (Chromium) and darktable only hit-test to coarse ancestors, so far-ancestor hits are accepted for those three apps. The category is stored per element (`hit`) and the policy is applied at build time (`build_shards.hit_ok`).

Near-duplicate boxes (IoU > 0.85) are merged. Icon-like elements without a name are kept in `uncaptioned` (no question) for later GPU captioning.

## Questions
- `name`: label as is, or "the "X" button", or with context ("the "Export" item in the File menu", "... in the "Print" dialog", "... on the "General" tab"). Used only when the label is unique on screen, or the context makes it unique.
- `position`: "the "X" button at the top-right of the screen" (disambiguates duplicate labels), and "the third icon from the left in the top toolbar". Ordinals are emitted only when the toolbar location is unique and the toolbar is a single row.
- `icon`: named icon-like elements.
- `function`: AT-SPI description/tooltip that differs from the name, or a DeepSeek paraphrase.
- `negative_hard`: a description that appears in another screenshot of the same app (another state) and is verified absent from this screenshot's FULL tree. Verification uses exact and fuzzy (ratio >= 80) name match and substring match against all names, descriptions and text contents, and rejects names made only of the sample-document vocabulary.
- `negative_easy`: the same from other apps.
- On the web, hard negatives come from other viewports or pages of the same site, verified against the current viewport's visible names and texts.

## Dedup against eval sets
Stage 1 is the repo's 64-bit dHash index (`dv/data/prep_v1.HashIndex`, Hamming <= 3). Stage 2 runs only on a stage-1 hit: it requires a 32x32 grayscale thumbnail mean absolute difference < 2.5/255 against an eval image. A re-encoded and resized eval image scores 0.2. Stage 1 alone fires on low-detail screens, e.g. a FreeCAD frame vs a Hacker News screenshot at 15/256 fine-hash bits. App overlap with ScreenSpot-Pro: vscode (we use VSCodium) and linux_common. Blender is installed but has no accessibility tree, so it is not used.

## Licensing
- Desktop screenshots show open-source applications (GPL/LGPL/MPL/MIT/Apache) rendering synthetic documents generated by `make_samples.py`, plus KiCad demo projects and FreeCAD test fixtures.
- Web screenshots are of third-party sites. They are for internal research only: do not redistribute them and do not make the dataset public.
- The web crawler respects robots.txt for `*` and `guisyn-research` and treats unreachable robots.txt as disallow. It makes at most 3 requests per site, never submits forms or types, and skips pages with password fields, login/paywall URLs, and adult or gambling domains.

## Run
```bash
# host setup (once): apt apps + xvfb/openbox/at-spi2/pyatspi, uv venv with --system-site-packages, playwright chromium,
# user `guisyn`, /data/guisyn/{raw,rawweb,homes} owned by guisyn, samples, eval hashes:
python make_samples.py --out /data/guisyn/samples
python /data/guisyn/repo/evals/gui_data.py --root /data/guisyn/evalgui && python evalhash.py --gui /data/guisyn/evalgui --out /data/guisyn/eval_hashes.json
# generation, capped at 18 cores with a systemd slice (the host is shared)
sudo systemd-run --unit=guisyn-desktop --slice=guisyn.slice --uid=friend --working-directory=/data/guisyn/code \
     /data/guisyn/venv/bin/python run_desktop.py --slots 16 --display_base 200 --hours 6 --steps 90
sudo systemd-run --unit=guisyn-web --slice=guisyn.slice --uid=guisyn --working-directory=/data/guisyn/code \
     -E PLAYWRIGHT_BROWSERS_PATH=/data/guisyn/pw bash -c 'for w in 0 1 2 3; do python web_worker.py --start $((15000+w*5000)) --count 5000 --wid 1$w & done; wait'
sudo systemctl set-property --runtime guisyn.slice CPUQuota=1800%
# after stopping web workers abruptly: python salvage_web.py  (closes their .tmp job files)
# cron: */30 * * * * /data/guisyn/code/cycle.sh   (build -> stats -> upload; logs in /data/guisyn/logs/)
touch /data/guisyn/STOP /data/guisyn/STOP_WEB                            # graceful stop
# QC
python qc.py --stage /data/guisyn/stage --out /data/guisyn/qc --source desktop --n 100
# training side
hf download Vision-Decision/guisyn-v0 --repo-type dataset --local-dir /content/guisyn-v0
python scripts/guisyn/to_v1.py --shards /content/guisyn-v0 --data_root /content/data --out /content/data/prep_v1 --extract
```

## v0 batch status (2026-11-18)
- First batch: 8,619 screenshots (5,741 desktop / 2,878 web), 492,549 questions, 18.8% negatives (73.6k hard / 19.2k easy), on HF as 25 files / 2.9 GB. Generation keeps running throttled; cron uploads every 30 min.
- QC on 100 random positives per source, judged on crops at native resolution, after the fixes below:
  - desktop: 1/100 errors (a hidden Qt combo-popup item; fixed by a rule for collapsed combo boxes);
  - web: 5/100 errors (1 wrong box, 1 invisible text, 3 noisy names).
  - No false negatives in about 100 negatives inspected over 10 frames.
- Earlier QC rounds found errors that the current filters remove (about 10-18% before):
  - LibreOffice: hidden toolbar items share a visible item's box, so the name and box get swapped. The full-tree same-box conflict check removes these.
  - LibreOffice: the GTK menubar reports extents about 20 px off. The hit test drops these.
  - Qt (Krita/QGIS): hidden popup widgets report SHOWING at the screen origin. The ancestor-SHOWING rule drops these.
  - Ordinals shifted when a toolbar item was filtered. Ordinals now require the toolbar's item set to equal the kept icons.
  - FreeCAD rich-text tooltips, icon-theme ids used as names, and "X icon" wording on text menus.
- Logs on the host (`/data/guisyn/logs/`): `cycle.log`, `build.log`, `upload.log` (one JSON per upload), `stats.log`, `run_desktop.log`, `report_v0.json`. Per-job worker logs are in `/data/guisyn/raw/logs/`.
- DeepSeek paraphrases: 3,601/4,000 element keys (`paraphrases_deepseek.json` in the dataset). They are applied to shards built after 04:32 UTC, and generic labels (OK/Close/...) are skipped.
