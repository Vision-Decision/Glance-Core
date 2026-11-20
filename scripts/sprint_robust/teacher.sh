#!/usr/bin/env bash
# Teacher pass for r5: the base generative Qwen3.5-4B grounds every shifted PRESENT training query of the aug pool;
# teacher_iou.json maps "image_id|text" -> IoU(teacher box, GT box) (0 if the teacher says none).
cd /content/dv
export PYTHONPATH=/content/dv TQDM_DISABLE=1
T=/content/out/sr/teacher; mkdir -p $T
python - <<'EOF'
import json, collections
by = collections.OrderedDict()
for l in open("/content/up/aug_natural.jsonl"):
    r = json.loads(l)
    for q in r["questions"]:
        if q.get("aug") in (None, "ulit") or not q["present"]:
            continue
        f = r["img"].split("/")[-1]
        rec = by.setdefault(r["image_id"], dict(image_id=r["image_id"], file=f, W=r["wh"][0], H=r["wh"][1], questions=[]))
        rec["questions"].append(dict(text=q["text"], present=True, box=q["box"], meta=dict(key=f"{r['image_id']}|{q['text']}", aug=q["aug"])))
with open("/content/out/sr/teacher/aug_present.jsonl", "w") as f:
    for r in by.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("teacher items", sum(len(r["questions"]) for r in by.values()))
EOF
python scripts/sprint_robust/base_gen.py --files $T/aug_present.jsonl --out_dir $T --bs 32 > $T/gen.log 2>&1
python - <<'EOF'
import gzip, json, collections
t, st = {}, collections.Counter()
for l in gzip.open("/content/out/sr/teacher/aug_present.jsonl.gz", "rt"):
    r = json.loads(l)
    t[r["meta"]["key"]] = round(r.get("iou", 0.0), 3)
    st[r["meta"]["aug"] + ("_ok" if r.get("iou", 0) >= 0.5 else ("_none" if r["box"] is None else "_wrong"))] += 1
json.dump(t, open("/content/up/teacher_iou.json", "w"))
print(json.dumps(dict(sorted(st.items())), indent=1))
EOF
echo "[teacher] DONE $(date -u)"
