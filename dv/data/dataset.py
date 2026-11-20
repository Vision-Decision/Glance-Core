"""Records -> model batches (shared image prefix + question blocks)."""
import json, math, random
import torch
from PIL import Image


def resize_dims(W, H, max_tokens, max_upscale, factor=32):
    """Largest (W',H'), multiples of `factor`, aspect ~kept, with (W'/f)*(H'/f) <= max_tokens."""
    s = min(math.sqrt(max_tokens * factor * factor / (W * H)), max_upscale)
    while True:
        w = max(factor, int(round(W * s / factor)) * factor)
        h = max(factor, int(round(H * s / factor)) * factor)
        if (w // factor) * (h // factor) <= max_tokens:
            return w, h
        s *= 0.98


class Tok:
    """Token-level prompt layout. Prefix: <vision_start> <image_pad>*n <vision_end>.
    Question block: <object_ref_start> {expr} <object_ref_end> <box_start>; read out at <box_start>."""

    def __init__(self, processor, max_q_tokens):
        self.tok = processor.tokenizer
        t = self.tok.convert_tokens_to_ids
        self.vs, self.ve, self.pad_img = t("<|vision_start|>"), t("<|vision_end|>"), t("<|image_pad|>")
        self.ors, self.ore, self.bs = t("<|object_ref_start|>"), t("<|object_ref_end|>"), t("<|box_start|>")
        for x in (self.vs, self.ve, self.pad_img, self.ors, self.ore, self.bs):
            assert isinstance(x, int) and x >= 0 and x != self.tok.unk_token_id, "special token missing"
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.max_q = max_q_tokens

    def question(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: self.max_q]
        return [self.ors] + ids + [self.ore, self.bs]


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_root, processor, cfg, limit=0, train=True):
        self.recs = [json.loads(l) for l in open(path)]
        if limit:
            self.recs = self.recs[:limit]
        self.root, self.proc, self.cfg, self.train = image_root, processor, cfg, train
        self.tok = Tok(processor, cfg.max_q_tokens)

    def __len__(self):
        return len(self.recs)

    def _pick(self, qs):
        """max_q_per_record requests; with min_orig_frac > 0, at least that share of them from original (orig=True)
        requests (Glance-X's sampler, for training on Glance-X's data mix)."""
        n, f = self.cfg.max_q_per_record, getattr(self.cfg, "min_orig_frac", 0.0)
        if f <= 0:
            return random.sample(qs, n)
        orig = [q for q in qs if q.get("orig")]
        aug = [q for q in qs if not q.get("orig")]
        n_orig = min(len(orig), max(int(round(n * f)), n - len(aug)))
        return random.sample(orig, n_orig) + random.sample(aug, min(len(aug), n - n_orig))

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(f"{self.root}/{r['file']}").convert("RGB")
        W, H = img.size
        w, h = resize_dims(W, H, self.cfg.max_vis_tokens, self.cfg.max_upscale)
        img = img.resize((w, h), Image.BICUBIC)
        pv = self.proc.image_processor(images=[img], do_resize=False, return_tensors="pt")
        qs = r["questions"]
        if self.train and len(qs) > self.cfg.max_q_per_record:
            qs = self._pick(qs)
        boxes = []
        for q in qs:
            b = q.get("box")
            boxes.append([b[0] / W, b[1] / H, b[2] / W, b[3] / H] if (b is not None) else [0, 0, 0, 0])
        return dict(pixel_values=pv["pixel_values"], grid_thw=pv["image_grid_thw"][0],
                    q_ids=[self.tok.question(q["text"]) for q in qs],
                    present=[bool(q["present"]) for q in qs], box=boxes, rec=r, qs=qs, orig_wh=(W, H))


def collate(items, tok: Tok, merge=2):
    B = len(items)
    grids = [it["grid_thw"] for it in items]
    n_img = [int(g[1] // merge) * int(g[2] // merge) for g in grids]
    Lp = max(n_img) + 2
    p_ids = torch.full((B, Lp), tok.pad, dtype=torch.long)
    p_valid = torch.zeros(B, Lp, dtype=torch.bool)
    p_img = torch.zeros(B, Lp, dtype=torch.bool)
    p_pos = torch.zeros(3, B, Lp, dtype=torch.long)
    q_start = []
    vis_valid = torch.zeros(B, max(n_img), dtype=torch.bool)
    centers = torch.zeros(B, max(n_img), 2)
    grid_hw = []
    for b, (g, n) in enumerate(zip(grids, n_img)):
        gh, gw = int(g[1] // merge), int(g[2] // merge)
        L = n + 2
        s = Lp - L  # left pad
        p_ids[b, s] = tok.vs
        p_ids[b, s + 1: s + 1 + n] = tok.pad_img
        p_ids[b, s + 1 + n] = tok.ve
        p_valid[b, s:] = True
        p_img[b, s + 1: s + 1 + n] = True
        ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        p_pos[0, b, s + 1: s + 1 + n] = 1
        p_pos[1, b, s + 1: s + 1 + n] = 1 + ii.flatten()
        p_pos[2, b, s + 1: s + 1 + n] = 1 + jj.flatten()
        end = 1 + max(gh, gw)
        p_pos[:, b, s + 1 + n] = end
        q_start.append(end + 1)
        vis_valid[b, :n] = True
        centers[b, :n, 0] = (jj.flatten() + 0.5) / gw
        centers[b, :n, 1] = (ii.flatten() + 0.5) / gh
        grid_hw.append((gh, gw))
    q_list, q_img = [], []
    for b, it in enumerate(items):
        for q in it["q_ids"]:
            q_list.append(q); q_img.append(b)
    M = len(q_list)
    Tq = max(len(q) for q in q_list)
    q_ids = torch.full((M, Tq), tok.pad, dtype=torch.long)
    q_valid = torch.zeros(M, Tq, dtype=torch.bool)
    q_last = torch.zeros(M, dtype=torch.long)
    q_pos = torch.zeros(3, M, Tq, dtype=torch.long)
    for m, (q, b) in enumerate(zip(q_list, q_img)):
        q_ids[m, :len(q)] = torch.tensor(q)
        q_valid[m, :len(q)] = True
        q_last[m] = len(q) - 1
        q_pos[:, m, :] = q_start[b] + torch.arange(Tq)
    q_img = torch.tensor(q_img)
    return dict(
        pixel_values=torch.cat([it["pixel_values"] for it in items]),
        grid_thw=torch.stack(grids),
        p_ids=p_ids, p_valid=p_valid, p_img_mask=p_img, p_pos=p_pos,
        q_ids=q_ids, q_valid=q_valid, q_last=q_last, q_pos=q_pos, q_img=q_img,
        vis_valid=vis_valid, centers=centers, grid_hw=torch.tensor(grid_hw)[q_img],
        present=torch.tensor(sum([it["present"] for it in items], [])),
        box=torch.tensor(sum([it["box"] for it in items], []), dtype=torch.float),
        n_tokens=int(p_valid.sum() + q_valid.sum()),
        meta=[dict(rec=it["rec"], qs=it["qs"], orig_wh=it["orig_wh"]) for it in items],
    )
