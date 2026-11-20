"""GroundCUA (MIT) without per-repo snapshot: list each app dir, pick N screenshots, download png+json with
exponential backoff on 429 and few workers.  python fetch_groundcua.py --per_app 80 --out DIR"""
import argparse, os, random, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
ap = argparse.ArgumentParser(); ap.add_argument("--per_app", type=int, default=80); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
api, rid = HfApi(), "ServiceNow/GroundCUA"
def retry(f, *x, **k):
    for i in range(12):
        try:
            return f(*x, **k)
        except Exception as e:
            if "429" in str(e) or "Too Many" in str(e) or "timed out" in str(e).lower() or "Connection" in str(e):
                time.sleep(min(300, 5 * 2 ** i) + random.random() * 5); continue
            raise
    return None
apps = [e.path.split("/", 1)[1] for e in retry(lambda: list(api.list_repo_tree(rid, "images", repo_type="dataset")))]
rng = random.Random(0); jobs = []
for app in apps:
    fs = retry(lambda: [e.path for e in api.list_repo_tree(rid, f"images/{app}", repo_type="dataset")]) or []
    rng.shuffle(fs)
    for p in fs[: a.per_app]:
        h = os.path.splitext(os.path.basename(p))[0]
        jobs += [p, f"data/{app}/{h}.json"]
    time.sleep(1)
print("apps", len(apps), "files", len(jobs), flush=True)
def get(p):
    if os.path.exists(os.path.join(a.out, p)):
        return 1
    r = retry(hf_hub_download, rid, p, repo_type="dataset", local_dir=a.out)
    return int(r is not None)
ok = 0
with ThreadPoolExecutor(a.workers) as ex:
    for i, r in enumerate(ex.map(get, jobs)):
        ok += r
        if i % 500 == 0:
            print("progress", i, ok, flush=True)
print("[groundcua] done", ok, "/", len(jobs), flush=True)
