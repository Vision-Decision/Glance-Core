"""v1 records -> model batches. One record = one *state* (image or text) + many typed question blocks.

Record (one jsonl line, written by dv/data/prep_v1.py):
  {"src": "...", "img": "coco/train2014/x.jpg" | null, "wh": [W, H] | null, "max_vis": 640,
   "text": "<state text>" | null, "plen": <estimated prefix tokens>,
   "questions": [
     {"type": "box",    "text": expr, "present": bool, "box": [x1,y1,x2,y2] px | null, "points": [[x,y],...] px | null, "neg": "hard"|"easy"|null},
     {"type": "choice", "instr": str, "options": [str,...], "target": [p,...]},
     {"type": "noul",   "instr": str, "target": p_yes},
     {"type": "score",  "instr": str, "options": [level str, ... lowest first], "target": [p,...]} ]}

Prompt layout (chat style; the image/text state is the shared prefix, each question block continues it):
  prefix : <|im_start|>user\n [<|vision_start|><|image_pad|>*n<|vision_end|>] [state text]
  box    : \n<|object_ref_start|>{expr}<|object_ref_end|><|im_end|>\n<|im_start|>assistant\n<|box_start|>
  choice : \n\n{instr}\n\nOptions:\nA. ..\nB. ..\nAnswer with the letter of the best option.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
  score  : same, "Scale (lowest to highest):" and "Answer with the letter of the level."
  noul   : \n\n{instr}\nAnswer yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
Readout = last token of the block (box: <|box_start|>; typed: the token that predicts the answer label).
"""
import json, math, random
import torch
from PIL import Image

from .dataset import resize_dims

LETTERS = [chr(ord("A") + i) for i in range(26)]
QTYPES = {"box": 0, "choice": 1, "noul": 2, "score": 3}
N_LAB = 28  # 0..25 = A..Z, 26 = no, 27 = yes


class ChatTok:
    def __init__(self, processor, cfg):
        self.tok = processor.tokenizer
        t = self.tok.convert_tokens_to_ids
        self.vs, self.ve, self.pad_img = t("<|vision_start|>"), t("<|vision_end|>"), t("<|image_pad|>")
        self.pad_vid = t("<|video_pad|>")
        self.ors, self.ore, self.bs = t("<|object_ref_start|>"), t("<|object_ref_end|>"), t("<|box_start|>")
        self.ims, self.ime = t("<|im_start|>"), t("<|im_end|>")
        for x in (self.vs, self.ve, self.pad_img, self.ors, self.ore, self.bs, self.ims, self.ime):
            assert isinstance(x, int) and x >= 0 and x != self.tok.unk_token_id, "special token missing"
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        self.enc = enc
        self.user_open = [self.ims] + enc("user\n")
        self.asst_typed = [self.ime] + enc("\n") + [self.ims] + enc("assistant\n<think>\n\n</think>\n\n")
        self.asst_box = [self.ime] + enc("\n") + [self.ims] + enc("assistant\n") + [self.bs]
        # label tokens (single tokens; asserted) -- groups: letters, no-variants, yes-variants
        def one(s):
            ids = enc(s)
            assert len(ids) == 1, (s, ids)
            return ids[0]
        self.letter_ids = [one(c) for c in LETTERS]
        self.no_ids = [one("No"), one("no")]
        self.yes_ids = [one("Yes"), one("yes")]
        self.max_state = cfg.max_state_tokens
        self.max_q_box = cfg.max_q_tokens
        self.max_q_typed = cfg.max_q_tokens_typed

    def label_token_groups(self):
        return [[i] for i in self.letter_ids] + [self.no_ids, self.yes_ids]

    def box_block(self, text):
        ids = self.enc(text)[: self.max_q_box]
        return self.enc("\n") + [self.ors] + ids + [self.ore] + self.asst_box

    def typed_block(self, q):
        t = q["type"]
        if t == "noul" and q.get("block") == "short":
            # scan readout: '\n<ref>{question}</ref><im_end>' (~question + 4 tokens instead of ~question + 18); read by the
            # same yes/no labels at the last token. Trained by self-distillation from the long block (scan/distill).
            ids = self.enc("\n") + [self.ors] + self.enc(q["instr"].strip())[: self.max_q_box] + [self.ore, self.ime]
            return ids, [26, 27]
        if t == "noul":
            body = "\n\n" + q["instr"].strip() + "\nAnswer yes or no."
            labs = [26, 27]
        else:
            opts = q["options"]
            head = "Options:" if t == "choice" else "Scale (lowest to highest):"
            tail = "Answer with the letter of the best option." if t == "choice" else "Answer with the letter of the level."
            body = "\n\n" + q["instr"].strip() + f"\n\n{head}\n" + "".join(
                f"{LETTERS[i]}. {str(o).strip()}\n" for i, o in enumerate(opts)) + tail
            labs = list(range(len(opts)))
        ids = self.enc(body)
        if len(ids) > self.max_q_typed:  # keep the end (options + answer instruction); prep drops most of these
            ids = ids[-self.max_q_typed:]
        return ids + self.asst_typed, labs

    def state_ids(self, text):
        return self.enc(text)[: self.max_state]


def _typed_target(q, smooth=0.0):
    if q["type"] == "noul":
        p = float(q["target"] if not isinstance(q["target"], list) else q["target"][0])
        return [1 - p, p]
    t = [float(x) for x in q["target"]]
    s = sum(t)
    t = [x / s for x in t] if s > 0 else [1 / len(t)] * len(t)
    if smooth > 0 and q["type"] == "choice" and max(t) >= 0.99:  # label smoothing only on one-hot choice targets
        t = [(1 - smooth) * x + smooth / len(t) for x in t]
    return t


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_root, processor, cfg, limit=0, train=True, recs=None):
        self.recs = recs if recs is not None else [json.loads(l) for p in path.split(",") for l in open(p)]
        if limit:
            self.recs = self.recs[:limit]
        self.root, self.proc, self.cfg, self.train = data_root, processor, cfg, train
        self.tok = ChatTok(processor, cfg)

    def __len__(self):
        return len(self.recs)

    def plen(self, i):
        return self.recs[i].get("plen", 512)

    def nq(self, i):
        return min(len(self.recs[i]["questions"]), self.cfg.max_q_per_record)

    def qtok(self, i):
        """rough padded block-token estimate of record i (chars/3 per typed block, 30 per box block)"""
        r = self.recs[i]
        if "_qtok" not in r:
            m = 0
            for q in r["questions"]:
                if q.get("type", "box") == "box":
                    m = max(m, 30)
                else:
                    m = max(m, min(self.cfg.max_q_tokens_typed, (len(q["instr"]) + sum(len(str(o)) + 4 for o in q.get("options", []))) // 3) + 30)
            r["_qtok"] = m * self.nq(i)
            r["_qmax"] = m
        return r["_qtok"]

    def kind(self, i):
        r = self.recs[i]
        return "video" if r.get("video") else ("img" if r.get("img") else "text")

    def sort_key(self, i):
        """image vs text (vs video) state first (their question blocks differ ~10x in length), then prefix-length bucket, then
        longest block -> little padding in both the prefix [B, Lp] and the question-block [M, Tq] tensors"""
        self.qtok(i)
        r = self.recs[i]
        return ({"img": 0, "text": 1, "video": 2}[self.kind(i)], self.plen(i) // 128, r["_qmax"] // 32, self.plen(i))

    def __getitem__(self, i):
        r = self.recs[i]
        cfg = self.cfg
        out = dict(rec=r)
        segs = [("ids", self.tok.user_open)]
        W = H = None
        if r.get("video"):  # Glance-Video: clip prefix (dv/data/video.py); typed questions only
            from .video import video_segs
            pv, g, vsegs = video_segs(r, self.root, self.proc, cfg, self.tok)
            out["pixel_values_videos"], out["video_grid_thw"] = pv, g
            segs = vsegs
        elif r.get("img"):
            img = Image.open(f"{self.root}/{r['img']}").convert("RGB")
            if r.get("crop"):  # eval crop-and-zoom: [x1,y1,x2,y2] px of the full image; outputs are in crop coords
                img = img.crop(tuple(int(round(v)) for v in r["crop"]))
            W, H = img.size
            w, h = resize_dims(W, H, r.get("max_vis") or cfg.max_vis_tokens, r.get("upscale", cfg.max_upscale))
            img = img.resize((w, h), Image.BICUBIC)
            pv = self.proc.image_processor(images=[img], do_resize=False, return_tensors="pt")
            out["pixel_values"], out["grid_thw"] = pv["pixel_values"], pv["image_grid_thw"][0]
            segs += [("ids", [self.tok.vs]), ("img", None), ("ids", [self.tok.ve])]
        if r.get("text"):
            segs.append(("ids", self.tok.state_ids(r["text"])))
        out["segs"] = segs
        qs = r["questions"]
        if self.train and len(qs) > cfg.max_q_per_record:
            if any(q.get("grp") for q in qs):  # description groups (sprint_robust): sample whole groups
                by = {}
                for q in qs:
                    by.setdefault(q.get("grp") or id(q), []).append(q)
                gs = list(by.values()); random.shuffle(gs)
                qs = []
                for g in gs:
                    if len(qs) + len(g) <= cfg.max_q_per_record:
                        qs += g
            else:
                qs = random.sample(qs, cfg.max_q_per_record)
        q_ids, qtype, labs, tgts, present, has_box, box, pts, gtb, ntg = [], [], [], [], [], [], [], [], [], []
        for q in qs:
            t = q.get("type", "box")
            qtype.append(QTYPES[t])
            if t == "box":
                q_ids.append(self.tok.box_block(q["text"]))
                labs.append([]); tgts.append([])
                pr = bool(q["present"])
                present.append(pr)
                # all referent boxes (v1.2 schema: "boxes" = 0, 1 or >=2 boxes; legacy single "box")
                bs = q.get("boxes") or ([q["box"]] if q.get("box") is not None else [])
                bs = [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in bs] if pr else []
                gtb.append(bs)
                has_box.append(len(bs) > 0)
                box.append(bs[0] if bs else [0, 0, 0, 0])
                p = q.get("points") or []
                pts.append([[x / W, y / H] for x, y in p] if (pr and not bs) else [])
                n = len(bs) or len(pts[-1])
                ntg.append(n if pr else 0)
            else:
                ids, lab = self.tok.typed_block(q)
                q_ids.append(ids); labs.append(lab)
                tgts.append(_typed_target(q, cfg.choice_smooth if self.train else 0.0))
                present.append(False); has_box.append(False); box.append([0, 0, 0, 0]); pts.append([])
                gtb.append([]); ntg.append(0)
        out.update(q_ids=q_ids, qtype=qtype, labs=labs, tgts=tgts, present=present, has_box=has_box, box=box,
                   points=pts, gt_boxes=gtb, n_tgt=ntg, qs=qs, orig_wh=(W, H), grp=[q.get("grp") for q in qs],
                   anchor=[bool(q.get("anchor")) for q in qs])
        return out


def collate_mixed(items, tok: ChatTok, merge=2, pad_to=64):
    B = len(items)
    grids, seq, n_img_l, ghw = [], [], [], []
    for it in items:
        if "grid_thw" in it:
            g = it["grid_thw"]; gh, gw = int(g[1] // merge), int(g[2] // merge)
        else:
            gh = gw = 0
        ghw.append((gh, gw))
        L = sum(len(s[1]) if s[0] == "ids" else (gh * gw if s[0] == "img" else s[1][0] * s[1][1]) for s in it["segs"])
        seq.append(L)
    Lp = -(-max(seq) // pad_to) * pad_to   # bucketed shapes: fla's Triton kernels autotune per new shape
    Nv = max(1, max(gh * gw for gh, gw in ghw))
    p_ids = torch.full((B, Lp), tok.pad, dtype=torch.long)
    p_valid = torch.zeros(B, Lp, dtype=torch.bool)
    p_img = torch.zeros(B, Lp, dtype=torch.bool)
    p_pos = torch.zeros(3, B, Lp, dtype=torch.long)
    vis_valid = torch.zeros(B, Nv, dtype=torch.bool)
    centers = torch.zeros(B, Nv, 2)
    q_start = []
    for b, it in enumerate(items):
        gh, gw = ghw[b]
        s = Lp - seq[b]
        p_valid[b, s:] = True
        cur = 0
        for kind, ids in it["segs"]:
            if kind == "ids":
                n = len(ids)
                p_ids[b, s:s + n] = torch.tensor(ids, dtype=torch.long)
                p_pos[:, b, s:s + n] = cur + torch.arange(n)
                s += n; cur += n
            elif kind == "vid":  # one temporal group of a video: same 3D positions as an image, no box targets
                vh, vw = ids
                n = vh * vw
                ii, jj = torch.meshgrid(torch.arange(vh), torch.arange(vw), indexing="ij")
                p_ids[b, s:s + n] = tok.pad_vid
                p_img[b, s:s + n] = True
                p_pos[0, b, s:s + n] = cur
                p_pos[1, b, s:s + n] = cur + ii.flatten()
                p_pos[2, b, s:s + n] = cur + jj.flatten()
                s += n; cur += max(vh, vw)
            else:
                n = gh * gw
                ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
                p_ids[b, s:s + n] = tok.pad_img
                p_img[b, s:s + n] = True
                p_pos[0, b, s:s + n] = cur
                p_pos[1, b, s:s + n] = cur + ii.flatten()
                p_pos[2, b, s:s + n] = cur + jj.flatten()
                vis_valid[b, :n] = True
                centers[b, :n, 0] = (jj.flatten() + 0.5) / gw
                centers[b, :n, 1] = (ii.flatten() + 0.5) / gh
                s += n; cur += max(gh, gw)
        q_start.append(cur)
    q_list, q_img = [], []
    for b, it in enumerate(items):
        for q in it["q_ids"]:
            q_list.append(q); q_img.append(b)
    M = len(q_list)
    Tq = -(-max(len(q) for q in q_list) // 16) * 16
    q_ids = torch.full((M, Tq), tok.pad, dtype=torch.long)
    q_valid = torch.zeros(M, Tq, dtype=torch.bool)
    q_last = torch.zeros(M, dtype=torch.long)
    q_pos = torch.zeros(3, M, Tq, dtype=torch.long)
    for m, (q, b) in enumerate(zip(q_list, q_img)):
        q_ids[m, :len(q)] = torch.tensor(q)
        q_valid[m, :len(q)] = True
        q_last[m] = len(q) - 1
        q_pos[:, m, :] = q_start[b] + torch.arange(Tq)
    cat = lambda k: sum([it[k] for it in items], [])
    labs, tgts = cat("labs"), cat("tgts")
    nl = max([len(l) for l in labs] + [1])
    lab_idx = torch.full((M, nl), -1, dtype=torch.long)
    tgt = torch.zeros(M, nl)
    for m, (l, t) in enumerate(zip(labs, tgts)):
        if l:
            lab_idx[m, :len(l)] = torch.tensor(l); tgt[m, :len(t)] = torch.tensor(t)
    # point targets -> flat list (m, x, y)
    pts = cat("points")
    pt_list = [(m, x, y) for m, p in enumerate(pts) for x, y in p]
    # all referent boxes -> flat list (m, x1, y1, x2, y2); items built before v1.2 fall back to the single box
    def _gtb(it):
        if "gt_boxes" in it and len(it["gt_boxes"]) == len(it["q_ids"]):
            return it["gt_boxes"]
        return [[b] if h else [] for b, h in zip(it["box"], it["has_box"])]
    gbs = sum([_gtb(it) for it in items], [])
    gb_list = [(m, *b) for m, bs in enumerate(gbs) for b in bs]
    n_tgt = sum([it["n_tgt"] if len(it.get("n_tgt", [])) == len(it["q_ids"]) else [len(b) for b in _gtb(it)]
                 for it in items], [])
    pv = [it["pixel_values"] for it in items if "pixel_values" in it]
    gt = [it["grid_thw"] for it in items if "grid_thw" in it]
    pvv = [it["pixel_values_videos"] for it in items if "pixel_values_videos" in it]
    vgt = [it["video_grid_thw"] for it in items if "video_grid_thw" in it]
    q_img = torch.tensor(q_img)
    grid_hw = torch.tensor(ghw)[q_img]
    gmap, grp = {}, []  # description-group ids, unique per (record, grp); -1 = none
    for b, it in enumerate(items):
        for g in it.get("grp", [None] * len(it["q_ids"])):
            grp.append(-1 if g is None else gmap.setdefault((b, g), len(gmap)))
    return dict(
        pixel_values=torch.cat(pv) if pv else None, grid_thw=torch.stack(gt) if gt else None,
        pixel_values_videos=torch.cat(pvv) if pvv else None, video_grid_thw=torch.stack(vgt) if vgt else None,
        p_ids=p_ids, p_valid=p_valid, p_img_mask=p_img, p_pos=p_pos,
        q_ids=q_ids, q_valid=q_valid, q_last=q_last, q_pos=q_pos, q_img=q_img,
        vis_valid=vis_valid, centers=centers, grid_hw=grid_hw,
        qtype=torch.tensor(cat("qtype")), lab_idx=lab_idx, tgt=tgt,
        present=torch.tensor(cat("present")), has_box=torch.tensor(cat("has_box")),
        box=torch.tensor(cat("box"), dtype=torch.float),
        pt_m=torch.tensor([p[0] for p in pt_list], dtype=torch.long),
        pt_xy=torch.tensor([[p[1], p[2]] for p in pt_list], dtype=torch.float).reshape(-1, 2),
        grp=torch.tensor(grp, dtype=torch.long),
        anchor=torch.tensor(sum([it.get("anchor", [False] * len(it["q_ids"])) for it in items], []), dtype=torch.bool),
        gb_m=torch.tensor([g[0] for g in gb_list], dtype=torch.long),
        gb_xyxy=torch.tensor([list(g[1:]) for g in gb_list], dtype=torch.float).reshape(-1, 4),
        n_tgt=torch.tensor(n_tgt, dtype=torch.long),
        n_tokens=int(p_valid.sum() + q_valid.sum()),
        meta=[dict(rec=it["rec"], qs=it["qs"], orig_wh=it["orig_wh"]) for it in items],
    )


class TokenBudgetSampler(torch.utils.data.Sampler):
    """Batches of records with padded prefix tokens (B * max plen) <= prefix_budget, total questions <= q_budget,
    B <= max_records. Records are length-sorted inside random pools so padding stays small; batch order shuffled."""

    def __init__(self, ds, prefix_budget, q_budget, max_records, seed=0, pool=8192, shuffle=True, qtok_budget=24576):
        self.ds, self.pb, self.qb, self.mr, self.tb = ds, prefix_budget, q_budget, max_records, qtok_budget
        self.seed, self.pool, self.shuffle, self.epoch = seed, pool, shuffle, 0
        self._batches = self._make()

    def _make(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.ds)))
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.pool):
            chunk = sorted(idx[s:s + self.pool], key=self.ds.sort_key)
            cur, mx, nq, nt = [], 0, 0, 0
            for i in chunk:
                pl, q, t = self.ds.plen(i), self.ds.nq(i), self.ds.qtok(i)
                if cur and ((len(cur) + 1) * max(mx, pl) > self.pb or nq + q > self.qb or len(cur) >= self.mr
                            or nt + t > self.tb or self._kind(i) != self._kind(cur[0])):
                    batches.append(cur); cur, mx, nq, nt = [], 0, 0, 0
                cur.append(i); mx = max(mx, pl); nq += q; nt += t
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _kind(self, i):
        r = self.ds.recs[i]
        return "video" if r.get("video") else ("img" if r.get("img") else "text")

    def set_epoch(self, e):
        self.epoch = e
        self._batches = self._make()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)
