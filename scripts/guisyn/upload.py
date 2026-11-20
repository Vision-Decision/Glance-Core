"""Incremental upload of completed shards to the PRIVATE HF dataset (one commit per cycle, only new files).

  python upload.py --stage /data/guisyn/stage --repo Vision-Decision/guisyn-v0
Only files already renamed into {stage} are considered (builders write .tmp elsewhere and rename when complete).
Files already on the Hub (by path) are skipped, so a failed/timed-out cycle is simply retried next time.
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="/data/guisyn/stage")
ap.add_argument("--repo", default="Vision-Decision/guisyn-v0")
ap.add_argument("--log", default="/data/guisyn/logs/upload.log")
ap.add_argument("--max_gb", type=float, default=12.0, help="cap per cycle (rest goes next cycle)")
a = ap.parse_args()
from huggingface_hub import HfApi
api = HfApi()
t0 = time.time()
res = dict(t=time.strftime("%F %T"), ok=False)
try:
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(a.repo, repo_type="dataset")
    if not info.private:
        raise SystemExit("refusing to upload: repo is not private")
    remote = set(api.list_repo_files(a.repo, repo_type="dataset"))
    local = []
    for dp, dn, fs in os.walk(a.stage):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fs:
            if f.endswith(".tmp") or f.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(dp, f), a.stage)
            local.append(rel)
    new = sorted(x for x in local if x not in remote or x in ("README.md", "stats.json"))
    # a shard = {name}.images.tar + .a11y.tar + .jsonl; send whole shards only, jsonl after its tars
    shards = {}
    for x in new:
        key = x.split(".")[0] if x.startswith("shards/") else x
        shards.setdefault(key, []).append(x)
    pick, size = [], 0
    for k in sorted(shards):
        fs = shards[k]
        if k.startswith("shards/") and not any(f.endswith(".jsonl") for f in fs) and \
                not any(f.endswith(".jsonl") for f in remote if f.startswith(k)):
            continue  # shard not complete yet (jsonl is renamed last)
        s = sum(os.path.getsize(os.path.join(a.stage, f)) for f in fs)
        if pick and size + s > a.max_gb * 1e9:
            break
        pick += fs; size += s
    res.update(n_new=len(new), n_pick=len(pick), gb=round(size / 1e9, 3))
    if pick:
        api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.stage, allow_patterns=pick,
                          commit_message=f"guisyn-v0: +{len(pick)} files ({size / 1e9:.2f} GB)")
    res["ok"] = True
except BaseException as e:
    res["error"] = repr(e)[:500]
res["secs"] = round(time.time() - t0)
with open(a.log, "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res))
