"""Download + normalize GUI grounding benchmarks into the repo's record format (one record per image).

Run on the VM:  python evals/gui_data.py --root /content/gui
Writes  {root}/{ss2,sspro,venus}.jsonl ; each line =
  {"file": <path relative to image_root>, "image_root": ..., "wh": [W,H],
   "questions": [{"text", "present", "box" (xyxy px or None), "meta": {...}}]}

ScreenSpot-v2 (OS-Copilot/ScreenSpot-v2): bbox is [x,y,w,h] -> converted to xyxy (checked against image size).
ScreenSpot-Pro (likaixin/ScreenSpot-Pro): bbox xyxy; English instruction, positive gt only (official run_ss_pro.sh).
VenusBench-GD (inclusionAI/VenusBench-GD): bbox xyxy; English `instruction`; refusal_grounding has no bbox
  (present=False). Subtask = file stem.
"""
import argparse, glob, json, os, zipfile
from collections import defaultdict
from PIL import Image


def write(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, "images", len(recs), "questions", sum(len(r["questions"]) for r in recs), flush=True)


def group(items, image_root):
    by = defaultdict(list)
    for file, q in items:
        by[file].append(q)
    recs = []
    for file, qs in by.items():
        W, H = Image.open(os.path.join(image_root, file)).size
        recs.append(dict(file=file, image_root=image_root, wh=[W, H], questions=qs))
    return recs


def ss2(root):
    from huggingface_hub import snapshot_download
    d = snapshot_download("OS-Copilot/ScreenSpot-v2", repo_type="dataset", local_dir=f"{root}/ss2_raw")
    img_root = f"{root}/ss2_raw/screenspotv2_image"
    if not os.path.isdir(img_root):
        zipfile.ZipFile(f"{d}/screenspotv2_image.zip").extractall(f"{root}/ss2_raw")
    items, bad = [], 0
    for plat in ["mobile", "desktop", "web"]:
        for i, a in enumerate(json.load(open(f"{d}/screenspot_{plat}_v2.json"))):
            x, y, w, h = a["bbox"]
            items.append((a["img_filename"], dict(text=a["instruction"], present=True, box=[x, y, x + w, y + h],
                          meta=dict(id=f"{plat}_{i}", platform=plat, ui_type=a["data_type"], source=a.get("data_source")))))
    recs = group(items, img_root)
    for r in recs:  # sanity: xywh interpretation must stay inside the image
        W, H = r["wh"]
        bad += sum(q["box"][2] > W + 2 or q["box"][3] > H + 2 for q in r["questions"])
    print("ss2 boxes outside image under xywh:", bad)
    write(f"{root}/ss2.jsonl", recs)


def sspro(root):
    from huggingface_hub import snapshot_download
    d = snapshot_download("likaixin/ScreenSpot-Pro", repo_type="dataset", local_dir=f"{root}/sspro_raw",
                          allow_patterns=["annotations/*", "images/**"])
    items = []
    for fn in sorted(glob.glob(f"{d}/annotations/*.json")):
        for a in json.load(open(fn)):
            items.append((a["img_filename"], dict(text=a["instruction"], present=True, box=a["bbox"],
                          meta=dict(id=a["id"], platform=a["platform"], application=a["application"],
                                    group=a["group"], ui_type=a["ui_type"]))))
    write(f"{root}/sspro.jsonl", group(items, f"{d}/images"))


def venus(root):
    from huggingface_hub import snapshot_download
    d = snapshot_download("inclusionAI/VenusBench-GD", repo_type="dataset", local_dir=f"{root}/venus_raw",
                          allow_patterns=["instruction/*", "images/**", "meta.json"])
    meta = {m["img_filename"]: m for m in json.load(open(f"{d}/meta.json"))}
    items = []
    for fn in sorted(glob.glob(f"{d}/instruction/*.json")):
        task = os.path.splitext(os.path.basename(fn))[0]
        for i, a in enumerate(json.load(open(fn))):
            m = meta.get(a["img_filename"], {})
            b = a.get("bbox")
            items.append((a["img_filename"], dict(text=a["instruction"], present=b is not None, box=b,
                          meta=dict(id=f"{task}_{i}", task=task, label=a.get("label"),
                                    platform=m.get("platform", a["img_filename"].split("/")[0]),
                                    application=m.get("application")))))
    write(f"{root}/venus.jsonl", group(items, f"{d}/images"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/gui")
    ap.add_argument("--only", default="ss2,sspro,venus")
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)
    for name in a.only.split(","):
        globals()[name](a.root)
