"""Debug: training-path forward (SharedPrefixLM + zoom batch) vs the inference engine (Lite) on the same items."""
import sys, os, json, torch
sys.path.insert(0, "/content/dv")
from dv.config import Config
from dv.model.build import build, load_ckpt
from arch.train.zoom_data import ZoomDataset, collate_zoom
from arch.train.train_zoom import to_dev
from dv.model.heads import box_iou_giou

ck = "/content/ckpt/glance-train-ckpts/v1.1/ckpt_final.pt"
sd = torch.load(ck, map_location="cpu", weights_only=False)
cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__}).update(dict(grad_ckpt=False, max_q_tokens=96))
net, heads, proc = build(cfg, "cuda"); load_ckpt(ck, net, heads); net.eval(); heads.eval()
ds = ZoomDataset("/content/data/prep_arch/smoke.jsonl", "/content/data", proc, cfg, p_miss=0.0)
items = [ds[i] for i in range(0, 60, 3)]
b = to_dev(collate_zoom(items, ds.tok), "cuda")
with torch.no_grad():
    feats = net.image_features(b["pixel_values"], b["grid_thw"])
    qf = net.image_features(b["crop_pixel_values"], b["crop_grid_thw"])
    print("n crop feats", [f.shape for f in qf], "mask sum", int(b["q_img_mask"].sum()))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hp, hq = net(b, feats, q_feats=qf)
    M, T, D = hq.shape
    hql = hq[torch.arange(M, device=hq.device), b["q_last"]]
    N = b["vis_valid"].shape[1]
    vis = hq.new_zeros(M, N, D); vis[b["vis_valid"]] = hq[b["q_img_mask"]]
    out = heads.box(hql, vis, b["vis_valid"], b["centers"], torch.arange(M, device=hq.device))
    iou, _ = box_iou_giou(out["box_pred"].float(), b["box"])
    print("acc50", float((iou >= 0.5).float().mean()), "center-in", float(((out["box_pred"][:, 0] + out["box_pred"][:, 2]) / 2 >= b["box"][:, 0]).float().mean()))
    for m in range(min(M, 6)):
        print(m, "present", bool(b["present"][m]), "p_logit", round(float(out["presence_logit"][m]), 2), "box", [round(x, 3) for x in out["box_pred"][m].tolist()],
              "tgt", [round(x, 3) for x in b["box"][m].tolist()], "iou", round(float(iou[m]), 3), "grid", b["grid_hw"][m].tolist())
    # plain first-pass on the same prefixes for sanity
    print("q_ids sample", b["q_ids"][0, :8].tolist(), "...", b["q_ids"][0, b["q_last"][0]-6:b["q_last"][0]+1].tolist())
    print("q_pos", b["q_pos"][:, 0, :3].tolist(), b["q_pos"][:, 0, b["q_last"][0]].tolist(), "p_pos last", b["p_pos"][:, 0, -1].tolist())
