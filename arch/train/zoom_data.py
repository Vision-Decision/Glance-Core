"""Training data for the in-block zoom step (phase 2, 'trained zoom').

From GUI records with pixel boxes (desktop / web / mobile subs of the v1.1 replay mixture): the shared prefix is the full
screenshot at a coarse budget; each question block is the 'pre' zoom layout of arch/zoom/run_zoom.py
    <vs><crop>*m<ve> \\n<ref>{q}</ref><im_end>\\n<im_start>assistant\\n<box_start>
(crop tokens causal, question tokens see the crop and each other). iso=True ('pre_alliso'): no block token may attend to
the full-image tokens of the prefix in the full-attention layers (the GDN state still carries the whole prefix); this was
the best zero-training layout (exp 2). The crop imitates first-pass placement: a window of
U(0.2, 0.33) of each side centred on the target +- U(-0.35, 0.35) of the window; with prob p_miss the window is moved off
the target (then the question is 'absent in this crop', teaching the gate to hand back to the first pass). Box targets are
in crop coordinates (box clipped to the crop; must keep >= 70% of its area, else the question is skipped).
"""
import json, random
import torch
from PIL import Image

from dv.data.dataset import resize_dims
from dv.data.mixed import ChatTok

Image.MAX_IMAGE_PIXELS = None


def image_pos(cur, gh, gw):
    ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    return torch.stack([torch.full((gh * gw,), cur), cur + ii.flatten(), cur + jj.flatten()]), \
        torch.stack([(jj.flatten() + 0.5) / gw, (ii.flatten() + 0.5) / gh], -1)


class ZoomDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_root, proc, cfg, subs=("desktop", "web", "mobile"), prefix_vis=2048, crop_vis=1024,
                 crop_up=2.0, q_per_rec=2, p_miss=0.0, seed=0, iso=True):
        self.recs = []
        for l in open(path):
            r = json.loads(l)
            if r.get("src") == "gui" and r.get("img") and any(r.get("sub", "").startswith(s) for s in subs):
                qs = [q for q in r["questions"] if q.get("type", "box") == "box" and q.get("present") and q.get("box")]
                if qs:
                    r["questions"] = qs; self.recs.append(r)
        self.root, self.proc, self.cfg = data_root, proc, cfg
        self.tok = ChatTok(proc, cfg)
        self.prefix_vis, self.crop_vis, self.crop_up, self.q_per_rec, self.p_miss = prefix_vis, crop_vis, crop_up, q_per_rec, p_miss
        self.iso = iso
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.recs)

    def crop_for(self, W, H, b, miss):
        rng = self.rng
        fx = fy = rng.uniform(0.2, 0.33)
        cw, ch = W * fx, H * fy
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if miss:  # move the window so that it does not contain the target centre
            for _ in range(10):
                x, y = rng.uniform(cw / 2, W - cw / 2), rng.uniform(ch / 2, H - ch / 2)
                if abs(x - cx) > cw / 2 + (b[2] - b[0]) / 2 or abs(y - cy) > ch / 2 + (b[3] - b[1]) / 2:
                    cx, cy = x, y; break
            else:
                return None
        else:
            cx += rng.uniform(-0.35, 0.35) * cw; cy += rng.uniform(-0.35, 0.35) * ch
        x1 = min(max(0, cx - cw / 2), W - cw); y1 = min(max(0, cy - ch / 2), H - ch)
        return [x1, y1, x1 + cw, y1 + ch]

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(f"{self.root}/{r['img']}").convert("RGB")
        W, H = img.size
        sx, sy = W / r["wh"][0], H / r["wh"][1]
        w, h = resize_dims(W, H, self.prefix_vis, 1.0)
        pv = self.proc.image_processor(images=[img.resize((w, h), Image.BICUBIC)], do_resize=False, return_tensors="pt")
        qs = self.rng.sample(r["questions"], min(self.q_per_rec, len(r["questions"])))
        blocks = []
        for q in qs:
            b = [q["box"][0] * sx, q["box"][1] * sy, q["box"][2] * sx, q["box"][3] * sy]
            miss = self.rng.random() < self.p_miss
            c = self.crop_for(W, H, b, miss)
            if c is None:
                continue
            cw, ch = c[2] - c[0], c[3] - c[1]
            nb = [max(b[0], c[0]) - c[0], max(b[1], c[1]) - c[1], min(b[2], c[2]) - c[0], min(b[3], c[3]) - c[1]]
            area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            inter = max(0, nb[2] - nb[0]) * max(0, nb[3] - nb[1])
            present = not miss and inter / area >= 0.7
            if not miss and not present:
                continue
            ci = img.crop(tuple(int(round(v)) for v in c))
            cw_, ch_ = resize_dims(ci.size[0], ci.size[1], self.crop_vis, self.crop_up)
            cpv = self.proc.image_processor(images=[ci.resize((cw_, ch_), Image.BICUBIC)], do_resize=False, return_tensors="pt")
            blocks.append(dict(text=q["text"], present=present, box=[nb[0] / cw, nb[1] / ch, nb[2] / cw, nb[3] / ch] if present else None,
                               pv=cpv["pixel_values"], grid=cpv["image_grid_thw"][0]))
        return dict(pv=pv["pixel_values"], grid=pv["image_grid_thw"][0], blocks=blocks, iso=self.iso)


def collate_zoom(items, tok, merge=2):
    """-> batch dict for SharedPrefixLM.forward(batch, img_feats, q_feats) + head targets in crop coordinates."""
    items = [it for it in items if it["blocks"]]
    B = len(items)
    pre = []
    for it in items:
        gh, gw = int(it["grid"][1]) // merge, int(it["grid"][2]) // merge
        ids = list(tok.user_open) + [tok.vs]
        pos = [torch.arange(len(ids)).repeat(3, 1)]
        cur = len(ids)
        ip, _ = image_pos(cur, gh, gw)
        ids += [tok.pad_img] * (gh * gw); pos.append(ip); cur += max(gh, gw)
        ids.append(tok.ve); pos.append(torch.tensor([[cur]] * 3)); cur += 1
        pre.append((ids, torch.cat(pos, 1), cur))
    Lp = max(len(p[0]) for p in pre)
    p_ids = torch.full((B, Lp), tok.pad, dtype=torch.long); p_valid = torch.zeros(B, Lp, dtype=torch.bool)
    p_img = torch.zeros(B, Lp, dtype=torch.bool); p_pos = torch.zeros(3, B, Lp, dtype=torch.long)
    for b, (ids, pos, _) in enumerate(pre):
        s = Lp - len(ids)
        p_ids[b, s:] = torch.tensor(ids); p_valid[b, s:] = True; p_pos[:, b, s:] = pos
        p_img[b, s:] = p_ids[b, s:] == tok.pad_img
    seqs, poss, kinds, q_img, cgrids, tgt_present, tgt_box, cpv, cgt = [], [], [], [], [], [], [], [], []
    for b, it in enumerate(items):
        q_start = pre[b][2]
        for bl in it["blocks"]:
            gh, gw = int(bl["grid"][1]) // merge, int(bl["grid"][2]) // merge
            q_ids = tok.enc(bl["text"])[: 96]
            tail = tok.enc("\n") + [tok.ors] + q_ids + [tok.ore] + tok.asst_box
            ip, _ = image_pos(q_start + 1, gh, gw)
            cur = q_start + 1 + max(gh, gw)
            ids = [tok.vs] + [tok.pad_img] * (gh * gw) + [tok.ve] + tail
            pos = torch.cat([torch.tensor([[q_start]] * 3), ip, (cur + torch.arange(1 + len(tail))).repeat(3, 1)], 1)
            seqs.append(ids); poss.append(pos); kinds.append(2 + gh * gw); q_img.append(b); cgrids.append((gh, gw))
            tgt_present.append(bl["present"]); tgt_box.append(bl["box"] or [0, 0, 0, 0])
            cpv.append(bl["pv"]); cgt.append(bl["grid"])
    M = len(seqs); T = -(-max(map(len, seqs)) // 16) * 16
    q_ids = torch.full((M, T), tok.pad, dtype=torch.long); q_valid = torch.zeros(M, T, dtype=torch.bool)
    q_pos = torch.zeros(3, M, T, dtype=torch.long); within = torch.zeros(M, T, T, dtype=torch.bool)
    q_last = torch.zeros(M, dtype=torch.long)
    for m, (ids, pos, nc) in enumerate(zip(seqs, poss, kinds)):
        L = len(ids)
        q_ids[m, :L] = torch.tensor(ids); q_valid[m, :L] = True; q_pos[:, m, :L] = pos
        q_pos[:, m, L:] = pos[:, -1:] + 1 + torch.arange(T - L)
        isc = torch.zeros(L, dtype=torch.bool); isc[:nc] = True
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        within[m, :L, :L] = (isc[:, None] & isc[None, :] & causal) | (~isc)[:, None]
        q_last[m] = L - 1
    within |= torch.eye(T, dtype=torch.bool)[None]
    Nv = max(gh * gw for gh, gw in cgrids)
    centers = torch.zeros(M, Nv, 2); vis_valid = torch.zeros(M, Nv, dtype=torch.bool)
    for m, (gh, gw) in enumerate(cgrids):
        _, c = image_pos(0, gh, gw); centers[m, :gh * gw] = c; vis_valid[m, :gh * gw] = True
    extra = {}
    if items and items[0].get("iso"):
        extra["q_pmask"] = ~p_img[torch.tensor(q_img)]
    return dict(**extra, pixel_values=torch.cat([it["pv"] for it in items]), grid_thw=torch.stack([it["grid"] for it in items]),
                crop_pixel_values=torch.cat(cpv), crop_grid_thw=torch.stack(cgt),
                p_ids=p_ids, p_valid=p_valid, p_img_mask=p_img, p_pos=p_pos, q_ids=q_ids, q_valid=q_valid, q_pos=q_pos,
                q_last=q_last, q_img=torch.tensor(q_img), within=within, q_img_mask=q_ids == tok.pad_img,
                centers=centers, vis_valid=vis_valid, grid_hw=torch.tensor(cgrids),
                present=torch.tensor(tgt_present), box=torch.tensor(tgt_box, dtype=torch.float), has_box=torch.tensor(tgt_present),
                n_tokens=int(p_valid.sum() + q_valid.sum()))
