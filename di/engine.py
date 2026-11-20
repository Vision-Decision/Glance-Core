"""Glance (decision-vision v1) as a Decision Index engine.

    python -m decision_index pipeline --engine di.engine:GlanceEngine --out runs/glance-v1
    decision-index vision pipeline --dir work/vision --engine di.engine:GlanceEngine --workers 1 --out runs/vision-glance-v1
    # weights default to the public release Vision-Decision/Glance-4B-preview @b6237b4 (adapter + heads + temperatures);
    # --option model=<repo or dir> --option revision=... for another release, --option ckpt=<.pt> for a training checkpoint

(the repository root must be on PYTHONPATH so that `di` and `dv` import.)

One request = one shared-prefix forward, exactly as in training/eval (dv/model/backbone.py):
  prefix : <|im_start|>user\n [<|vision_start|><|image_pad|>*n<|vision_end|>]*k [state text]
  block  : one per question (and per option chunk, see below), the dv/data/mixed.py typed layout:
           \n\n{instructions}\n\nOptions:\nA. ..\nB. ..\nAnswer with the letter of the best option.<|im_end|>\n
           <|im_start|>assistant\n<think>\n\n</think>\n\n            (noul: "\nAnswer yes or no.")
  readout: dv/model/typed.py TypedHead at the block's last token (LM-head rows of A..Z / No,Yes + trained residual),
           divided by the post-hoc temperature of the question type, softmax over the block's labels.

Fixed rendering for every benchmark (no prompt tuning):
  * state: a string as-is; any other non-empty value as JSON (json.dumps, ensure_ascii=False) -- the eval.py od_to_v1 rule.
  * instructions: a string as-is, otherwise JSON.
  * option line i: "{letter}. {description}" when the description equals the key or is null, else
    "{letter}. {key}: {description}" (od_to_v1 rule). Rows with prompt.options_in_image (MMMU-Pro vision) show bare letters.
  * noul: instructions + "\n(yes = <criteria.true>; no = <criteria.false>)" when criteria are given (od_to_v1 rule).
  * images: native resolution, never upscaled, capped at max_vis_tokens 32x32 tokens (default 1600 = 1,638,400 px,
    the board's 1.6 MP cap; v1 trained at <= 640 tokens and the board asks to lift author caps up to 1.6 MP).

Nothing is truncated (the training-time caps max_state_tokens / max_q_tokens_typed are NOT applied). A request whose
prefix or a single block exceeds the declared limits (max_prefix_tokens, max_block_tokens) raises Unsupported.

LARGE-OPTION FALLBACK (> 26 options; the typed readout has 26 letter labels): option (c) chunking, see di/LARGE_OPTIONS.md.
  The options are split, in their original order, into ceil(N/26) near-equal chunks; each chunk is its own block on the
  same prefix (same forward). chunk_mode:
    "tournament" (default): a second forward with one block holding every chunk's top-k options (k = min(advance=3,
        26 // chunks), original order). M_c = final-round mass of chunk c's finalists; a non-finalist o of chunk c gets
        M_c * P_chunk(o); the finalists share the rest in final-round proportions (so the final read decides the
        argmax). Every option is shown to the model and gets a probability. Same shape as StartLux v1.1's group rounds.
    "concat": one forward; global softmax over the concatenated chunk logits (cheaper, less calibrated).
  This is a stop-gap and is declared in provenance/environment.json.

Determinism: one request at a time, fixed block grouping, deterministic torch flags, bf16 autocast on CUDA, no sampling.
fla's Triton kernels pick a config by autotuning per shape; prefix/block lengths are padded to multiples of 64/16 (as in
eval) so the set of shapes is small. Answers are argmaxes, so tie-level numeric noise is the only source of disagreement.
"""
import json
import math
import os
import time

import torch

try:
    from decision_index.engines import Engine, Unsupported
except ImportError:  # the kit is not installed: a minimal stand-in so the engine can be used directly
    class Unsupported(ValueError):
        pass

    class Engine:
        name = "engine"
        provenance = {}
        latency = ""

        def __init__(self, **options):
            self.options = options

        def warmup(self):
            for _ in range(2):
                self("The color is red.", {"warmup": {"type": "choice", "instructions": "Which color is named?",
                                                      "criteria": {"red": "red", "blue": "blue"}}})

        def runtime(self):
            return {}

        def synchronize(self):
            pass

        def close(self):
            pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_LETTERS = 26
LETTERS = [chr(ord("A") + i) for i in range(N_LETTERS)]
LAB_NO, LAB_YES = 26, 27
CAPACITY = "maximum context length"   # http-engine capacity markers, so a server wrapper maps these to `unsupported`


def as_text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def is_empty(x):
    return x is None or x == "" or (isinstance(x, (dict, list)) and not x)


def option_line(letter, key, desc, bare=False):
    if bare:
        return f"{letter}. {key}"
    if desc is None or as_text(desc).strip() == str(key).strip():
        return f"{letter}. {str(key).strip()}"
    return f"{letter}. {str(key).strip()}: {as_text(desc).strip()}"


def chunk_sizes(n, cap=N_LETTERS):
    k = -(-n // cap)
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


class GlanceEngine(Engine):
    name = "glance-v1"
    latency = ("In-process, device-synchronized request wall time: tokenization, image decode/resize, vision tower, one "
               "shared-prefix forward (two for >26-option tournament fallback), readout; excludes model loading.")

    def __init__(self, model="Vision-Decision/Glance-4B-preview", revision="b6237b4a5856c1e44714b1204163b62a36832851",
                 ckpt=None, temps=None, device=None, dtype=None, model_id=None, max_vis_tokens=1600,
                 max_prefix_tokens=65536, max_block_tokens=32768, group_tokens=32768, chunk_mode="tournament", advance=3,
                 image_root=None, model_name="glance-v1", deterministic=True, **options):
        """Weights: by default the public release `model` (HF repo id or a local dir with adapter_model.safetensors,
        heads.safetensors, glance_config.json), pinned to `revision`. A private repo works the same way with HF_TOKEN.
        `ckpt=` (a training checkpoint .pt) overrides it; `temps=` (eval_results.json) overrides the release temperatures."""
        super().__init__(**options)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or ("bfloat16" if self.device == "cuda" else "float32")
        assert self.dtype in ("bfloat16", "float32"), self.dtype
        assert chunk_mode in ("tournament", "concat"), chunk_mode
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.manual_seed(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        import sys
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from dv.config import Config
        from dv.data.mixed import ChatTok
        from dv.model.build import build, load_ckpt

        self.temps = {}
        if ckpt:
            ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(REPO, ckpt)
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            cfg = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
            del sd
            source = {"checkpoint": ckpt, "checkpoint_bytes": os.path.getsize(ckpt)}
        else:
            files = self._release_files(model, revision)
            g = json.load(open(files["glance_config.json"]))
            cfg = Config().update(dict(
                task="v1", model_id=g["base_model"], attn_mask=g["attn_mask"], lora_r=g["lora"]["r"],
                lora_alpha=g["lora"]["lora_alpha"], lora_target=g["lora"]["target_modules"],
                max_vis_tokens=g["max_vis_tokens"], max_upscale=g["max_upscale"], max_q_tokens=g["max_q_tokens"],
                max_q_tokens_typed=g["max_q_tokens_typed"], max_state_tokens=g["max_state_tokens"], **g["heads"]))
            self.temps = dict(g.get("temperatures") or {})
            source = {"repo": model, "revision": revision, "release": g.get("name"), "version": g.get("version"),
                      "train_step": g.get("train_step")}
        if model_id:
            cfg = cfg.update(dict(model_id=model_id))
        net, heads, proc = build(cfg, self.device, lora=True)
        if ckpt:
            load_ckpt(ckpt, net, heads)
        else:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            res = set_peft_model_state_dict(net.hf, load_file(files["adapter_model.safetensors"]))
            bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
            assert not bad, f"unexpected LoRA keys: {bad[:5]}"
            heads.load_state_dict(load_file(files["heads.safetensors"]))
        if self.dtype == "float32":
            net.hf.float()
        heads.float()
        net.eval(); heads.eval()
        self.net, self.heads, self.proc, self.cfg = net, heads, proc, cfg
        self.tok = ChatTok(proc, cfg)
        if temps:
            tp = temps if os.path.isabs(temps) else os.path.join(REPO, temps)
            d = json.load(open(tp))
            self.temps = d.get("temps", d)
        self.max_vis, self.max_prefix, self.max_block = int(max_vis_tokens), int(max_prefix_tokens), int(max_block_tokens)
        self.group_tokens, self.chunk_mode, self.image_root, self.model_name = int(group_tokens), chunk_mode, image_root, model_name
        self.advance = int(advance)
        self.provenance = {
            "kind": "in-process", "engine": "di.engine:GlanceEngine", "base": cfg.model_id, **source,
            "device": self.device, "dtype": self.dtype,
            "temperatures": self.temps, "attn_mask": cfg.attn_mask,
            "max_vis_tokens": self.max_vis, "max_pixels": self.max_vis * 1024, "image_resize": "native, never upscaled; "
            "downscaled (bicubic, multiples of 32 px) only when over max_vis_tokens 32x32 tokens",
            "declared_limits": {"options_per_choice": 255, "max_prefix_tokens": self.max_prefix,
                                "max_block_tokens": self.max_block},
            "large_option_fallback": {"applies_to": "> 26 options", "mode": chunk_mode,
                                      "advance": int(advance),
                                      "rule": "original-order near-equal chunks of <= 26 on the shared prefix; tournament: "
                                              "the top-k of every chunk (k = min(advance, 26 // chunks)) go, in original order, "
                                              "to a second forward; finalists share 1 - r in final-round proportions, "
                                              "other options get M_c * P_chunk(o) (M_c = final mass of chunk c's finalists, "
                                              "r = sum_c M_c * (1 - P_chunk(finalists of c)))"},
            "policy": "One fixed rendering for every benchmark (di/engine.py docstring); no truncation, no option "
                      "filtering; over-limit requests raise Unsupported.",
        }

    @staticmethod
    def _release_files(model, revision):
        names = ("glance_config.json", "adapter_model.safetensors", "heads.safetensors")
        if os.path.isdir(model):
            return {n: os.path.join(model, n) for n in names}
        from huggingface_hub import hf_hub_download
        return {n: hf_hub_download(model, n, revision=revision) for n in names}

    # ------------------------------------------------------------------ kit hooks
    def runtime(self):
        info = {"torch": torch.__version__, "device": self.device}
        if self.device == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        try:
            import transformers
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        try:
            import fla
            info["fla"] = getattr(fla, "__version__", "?")
        except Exception:
            info["fla"] = None
        return info

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    # ------------------------------------------------------------------ rendering
    def _load_images(self, images):
        from PIL import Image
        import io
        from dv.data.dataset import resize_dims
        try:
            from decision_index.vision.images import load
        except ImportError:
            load = lambda ref, root=None: open(os.path.join(root or "", ref), "rb").read() if isinstance(ref, str) else ref
        out, pix = [], []
        for ref in images or ():
            im = Image.open(io.BytesIO(load(ref, self.image_root))).convert("RGB")
            W, H = im.size
            w, h = resize_dims(W, H, self.max_vis, 1.0)
            if (w, h) != (W, H):
                im = im.resize((w, h), Image.BICUBIC)
            out.append(im)
            pix.append(w * h)
        return out, pix

    def _blocks(self, questions, prompt):
        """-> list of (qkey, kind, ids, labels, keys); kind in choice | noul | chunk:<i>."""
        enc, tail = self.tok.enc, self.tok.asst_typed
        bare = bool((prompt or {}).get("options_in_image"))
        blocks = []
        for qk, q in questions.items():
            instr = as_text(q.get("instructions", "")).strip()
            if q["type"] == "noul":
                crit = q.get("criteria") or {}
                if crit:
                    instr += f"\n(yes = {as_text(crit.get('true') or 'true')}; no = {as_text(crit.get('false') or 'false')})"
                blocks.append((qk, "noul", enc("\n\n" + instr + "\nAnswer yes or no.") + tail, [LAB_NO, LAB_YES], None))
                continue
            if q["type"] != "choice":
                raise Unsupported(f"question type {q['type']!r} is not supported")
            items = list(q["criteria"].items())
            if len(items) < 2:
                raise Unsupported("a choice needs at least two options")
            if len(items) > 255:
                raise Unsupported(f"more than 255 options per choice ({len(items)})")
            s = 0
            sizes = chunk_sizes(len(items))
            for ci, n in enumerate(sizes):
                part = items[s:s + n]
                body = "\n\n" + instr + "\n\nOptions:\n" + "".join(
                    option_line(LETTERS[i], k, d, bare) + "\n" for i, (k, d) in enumerate(part)) + \
                    "Answer with the letter of the best option."
                kind = "choice" if len(sizes) == 1 else f"chunk:{ci}"
                blocks.append((qk, kind, enc(body) + tail, list(range(n)), [k for k, _ in part]))
                s += n
        return blocks

    # ------------------------------------------------------------------ forward
    def _prefix(self, state, imgs):
        segs = [("ids", self.tok.user_open)]
        pv = None
        if imgs:
            p = self.proc.image_processor(images=imgs, do_resize=False, return_tensors="pt")
            pv = (p["pixel_values"], p["image_grid_thw"])
            for g in p["image_grid_thw"]:
                segs += [("ids", [self.tok.vs]), ("img", (int(g[1]) // 2, int(g[2]) // 2)), ("ids", [self.tok.ve])]
        if not is_empty(state):
            segs.append(("ids", self.tok.enc(as_text(state))))
        return segs, pv

    def _batch(self, segs, q_list, pad_to=64):
        """B=1 generalisation of dv.data.mixed.collate_mixed (k images per prefix); identical tensors for k <= 1."""
        tok = self.tok
        L = sum(len(x) if k == "ids" else x[0] * x[1] for k, x in segs)
        Lp = -(-L // pad_to) * pad_to
        p_ids = torch.full((1, Lp), tok.pad, dtype=torch.long)
        p_valid = torch.zeros(1, Lp, dtype=torch.bool)
        p_img = torch.zeros(1, Lp, dtype=torch.bool)
        p_pos = torch.zeros(3, 1, Lp, dtype=torch.long)
        s, cur = Lp - L, 0
        p_valid[0, s:] = True
        for kind, x in segs:
            if kind == "ids":
                n = len(x)
                p_ids[0, s:s + n] = torch.tensor(x, dtype=torch.long)
                p_pos[:, 0, s:s + n] = cur + torch.arange(n)
                s += n; cur += n
            else:
                gh, gw = x
                n = gh * gw
                ii, jj = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
                p_ids[0, s:s + n] = tok.pad_img
                p_img[0, s:s + n] = True
                p_pos[0, 0, s:s + n] = cur
                p_pos[1, 0, s:s + n] = cur + ii.flatten()
                p_pos[2, 0, s:s + n] = cur + jj.flatten()
                s += n; cur += max(gh, gw)
        M = len(q_list)
        Tq = -(-max(len(q) for q in q_list) // 16) * 16
        q_ids = torch.full((M, Tq), tok.pad, dtype=torch.long)
        q_valid = torch.zeros(M, Tq, dtype=torch.bool)
        q_last = torch.zeros(M, dtype=torch.long)
        for m, q in enumerate(q_list):
            q_ids[m, :len(q)] = torch.tensor(q)
            q_valid[m, :len(q)] = True
            q_last[m] = len(q) - 1
        q_pos = (cur + torch.arange(Tq))[None, None, :].expand(3, M, Tq).clone()
        return dict(p_ids=p_ids, p_valid=p_valid, p_img_mask=p_img, p_pos=p_pos, q_ids=q_ids, q_valid=q_valid,
                    q_last=q_last, q_pos=q_pos, q_img=torch.zeros(M, dtype=torch.long)), L

    def _groups(self, lens):
        """Deterministic grouping of blocks so that M * Tq_padded <= group_tokens (blocks sorted by length)."""
        order = sorted(range(len(lens)), key=lambda i: (lens[i], i))
        groups, cur, mx = [], [], 0
        for i in order:
            t = -(-lens[i] // 16) * 16
            if cur and (len(cur) + 1) * max(mx, t) > self.group_tokens:
                groups.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, t)
        if cur:
            groups.append(cur)
        return groups

    @torch.no_grad()
    def _forward(self, segs, pv, q_list):
        """-> [M, 28] fp32 label logits (TypedHead, before temperature), number of forwards."""
        dev = self.device
        feats = None
        if pv is not None:
            feats = self.net.image_features(pv[0].to(dev), pv[1].to(dev))
        out = [None] * len(q_list)
        groups = self._groups([len(q) for q in q_list])
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (dev == "cuda" and self.dtype == "bfloat16") \
            else torch.autocast("cpu", enabled=False)
        for g in groups:
            batch, _ = self._batch(segs, [q_list[i] for i in g])
            batch = {k: v.to(dev) for k, v in batch.items()}
            with ctx:
                _, hq = self.net(batch, feats or [])
            h = hq[torch.arange(len(g), device=dev), batch["q_last"]]
            lab = self.heads.typed(h).float().cpu()
            for j, i in enumerate(g):
                out[i] = lab[j]
        return torch.stack(out), len(groups)

    def _probs(self, lab, labels, kind):
        T = self.temps.get(kind, 1.0)
        x = lab[labels].double() / T
        return x.softmax(-1)

    # ------------------------------------------------------------------ call
    def __call__(self, state, questions, images=(), prompt=None):
        t0 = time.perf_counter()
        imgs, pix = self._load_images(images)
        segs, pv = self._prefix(state, imgs)
        L = sum(len(x) if k == "ids" else x[0] * x[1] for k, x in segs)
        if L > self.max_prefix:
            raise Unsupported(f"{CAPACITY}: prefix of {L} tokens exceeds the declared {self.max_prefix}")
        blocks = self._blocks(questions, prompt)
        longest = max(len(b[2]) for b in blocks)
        if longest > self.max_block:
            raise Unsupported(f"{CAPACITY}: question block of {longest} tokens exceeds the declared {self.max_block}")
        lab, n_fwd = self._forward(segs, pv, [b[2] for b in blocks])
        answers, chunked = {}, {}
        by_q = {}
        for b, lg in zip(blocks, lab):
            by_q.setdefault(b[0], []).append((b, lg))
        final_jobs = []
        for qk, q in questions.items():
            parts = by_q[qk]
            if q["type"] == "noul":
                p = self._probs(parts[0][1], parts[0][0][3], "noul")
                answers[qk] = {"type": "noul", "noul": float(p[1])}
                continue
            keys = list(q["criteria"])
            if len(parts) == 1:
                p = self._probs(parts[0][1], parts[0][0][3], "choice")
                answers[qk] = self._answer(keys, p.tolist())
                continue
            chunked[qk] = len(parts)
            if self.chunk_mode == "concat":
                x = torch.cat([lg[b[3]].double() for b, lg in parts]) / self.temps.get("choice", 1.0)
                answers[qk] = self._answer(keys, x.softmax(-1).tolist())
            else:
                local = [self._probs(lg, b[3], "choice") for b, lg in parts]
                k = max(1, min(self.advance, N_LETTERS // len(parts)))
                fin = []  # per chunk: local indices of its top-k (by prob, ties -> earlier option)
                for p in local:
                    order = sorted(range(len(p)), key=lambda i: (-float(p[i]), i))
                    fin.append(sorted(order[:k]))
                winners = [b[4][i] for (b, _), f in zip(parts, fin) for i in f]  # original order
                final_jobs.append((qk, q, parts, local, winners, fin))
        if final_jobs:
            enc, tail = self.tok.enc, self.tok.asst_typed
            bare = bool((prompt or {}).get("options_in_image"))
            q_list = []
            for qk, q, parts, local, winners, fin in final_jobs:
                instr = as_text(q.get("instructions", "")).strip()
                body = "\n\n" + instr + "\n\nOptions:\n" + "".join(
                    option_line(LETTERS[i], k, q["criteria"][k], bare) + "\n" for i, k in enumerate(winners)) + \
                    "Answer with the letter of the best option."
                q_list.append(enc(body) + tail)
            lab2, n2 = self._forward(segs, pv, q_list)
            n_fwd += n2
            for (qk, q, parts, local, winners, fin), lg in zip(final_jobs, lab2):
                pf = self._probs(lg, list(range(len(winners))), "choice").tolist()
                # chunk c's mass M_c = final mass of its finalists; non-finalists of c get M_c * p_c(o); finalists share
                # the remaining 1 - r in the final round's proportions, so the final round decides the argmax.
                probs, j, parts_nf, r = [], 0, [], 0.0
                for c, (p, f) in enumerate(zip(local, fin)):
                    Mc = sum(pf[j:j + len(f)])
                    cov = float(sum(p[i] for i in f))
                    parts_nf.append((Mc, f))
                    r += Mc * (1 - cov)
                    j += len(f)
                j = 0
                for c, (p, (Mc, f)) in enumerate(zip(local, parts_nf)):
                    fs = set(f)
                    for i in range(len(p)):
                        if i in fs:
                            probs.append(pf[j + f.index(i)] * (1 - r))
                        else:
                            probs.append(Mc * float(p[i]))
                    j += len(f)
                answers[qk] = self._answer(list(q["criteria"]), probs)
        self.synchronize()
        raw = {"prefix_tokens": L, "blocks": len(blocks), "max_block_tokens": longest, "forwards": n_fwd,
               "chunked": chunked, "ms": (time.perf_counter() - t0) * 1000}
        if images:
            raw["image_pixels"] = pix
            raw["image_tokens"] = sum(x[0] * x[1] for k, x in segs if k == "img")
        return {"model": self.model_name, "answers": answers, "usage": {"input_tokens": L + sum(len(b[2]) for b in blocks)}}, raw

    @staticmethod
    def _answer(keys, probs):
        s = sum(probs)
        probs = [p / s for p in probs]
        best = max(range(len(keys)), key=lambda i: (probs[i], -i))
        return {"type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, probs))}
