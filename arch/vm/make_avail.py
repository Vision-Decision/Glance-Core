"""Filter the v1.1 replay mixture to records whose image exists on this VM (identical rule on every VM; the excluded
sources are listed so that runs on different VMs can be checked to use the same records)."""
import json, os, sys, collections, hashlib
from huggingface_hub import hf_hub_download
p = hf_hub_download("Vision-Decision/glance-train-data", "v1.2/train_base.jsonl", repo_type="dataset", local_dir="/content/gtd")
skip_src = set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else set()
os.makedirs("/content/data/prep_arch", exist_ok=True)
n = k = 0; miss = collections.Counter(); h = hashlib.sha1()
with open("/content/data/prep_arch/train_base_avail.jsonl", "w") as f:
    for l in open(p):
        r = json.loads(l); n += 1
        sub = f"{r.get('src')}/{r.get('sub')}"
        if sub in skip_src or (r.get("img") and not os.path.exists(f"/content/data/{r['img']}")):
            miss[sub] += 1; continue
        f.write(l); k += 1; h.update(l.encode())
print(json.dumps(dict(records=n, kept=k, sha1=h.hexdigest(), missing=dict(miss))), flush=True)
