"""Per-benchmark shape statistics of the public Decision Index suites, measured with OUR tokenizer and OUR rendering
(di/engine.py), so the gap analysis (di/GAPS.md) is about what Glance would actually see.

    python -m di.suite_stats --suite suite-0.3 --vision work/vision --out di/suite_stats.json

Per benchmark: requests, questions per request, option-count distribution (and requests with a >26-option question),
prefix tokens (state, + image tokens at the 1.6 MP cap), longest question block, total tokens per request
(prefix + all blocks), and an estimated latency per request on one RTX PRO 6000 from a simple linear model
(fixed overhead + tokens / throughput, both CLI-tunable; calibrate with di/latency_probe.py on the GPU).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class Renderer:
    """The engine's rendering without the model: tokenizer + ChatTok only."""

    def __init__(self, model_id="Qwen/Qwen3.5-4B"):
        from transformers import AutoProcessor
        from dv.config import Config
        from dv.data.mixed import ChatTok
        from di import engine as E

        self.E = E
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.tok = ChatTok(self.proc, Config())

        class Stub:  # bind the engine's pure rendering methods to this object
            pass
        self.stub = Stub()
        self.stub.tok = self.tok

    def blocks(self, questions, prompt=None):
        return self.E.GlanceEngine._blocks(self.stub, questions, prompt)

    def state_tokens(self, state):
        n = len(self.tok.user_open)
        if not self.E.is_empty(state):
            n += len(self.tok.enc(self.E.as_text(state)))
        return n


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def latency_ms(prefix, blocks_tokens, n_blocks, img_tokens, a, chunked):
    """Linear model: fixed + (prefix + sum(blocks)) / tok_per_s + per-block + vision tower; x2 forward for tournament."""
    t = a.fixed_ms + (prefix + blocks_tokens) / a.tok_per_s * 1000 + n_blocks * a.per_block_ms + img_tokens * a.vis_ms_per_tok
    if chunked:
        t += a.fixed_ms * 0.6 + prefix / a.tok_per_s * 1000
    return t


def text_stats(suite_dir, edition, R, a, rows=None):
    from decision_index.suite.io import Suite, read_jsonl
    from decision_index import editions

    if rows:  # raw row files (e.g. straight from `suite rebuild`), filtered with the edition's read-time rules
        paths, keep = rows, editions.in_edition(edition)
    else:
        suite = Suite(suite_dir, edition)
        paths, keep = suite.row_paths, suite.in_edition
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        for row in read_jsonl(path):
            e = row["_evaluation"]
            if not keep(e):
                continue
            b = e.get("dataset") or row.get("family")
            d = per[b]
            qs = row["questions"]
            blocks = R.blocks(qs)
            st = R.state_tokens(row["state"])
            bl = [len(x[2]) for x in blocks]
            nopt = [len(q.get("criteria") or {}) if q["type"] == "choice" else 2 for q in qs.values()]
            chunked = any(n > 26 for n in nopt)
            d["prefix"].append(st)
            d["block_max"].append(max(bl))
            d["total"].append(st + sum(bl))
            d["nq"].append(len(qs))
            d["opt_max"].append(max(nopt))
            d["types"].extend(q["type"] for q in qs.values())
            d["over26"].append(chunked)
            d["lat"].append(latency_ms(st, sum(bl), len(bl), 0, a, chunked))
    return summarize(per, a)


def vision_stats(vdir, R, a):
    from PIL import Image
    from dv.data.dataset import resize_dims

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(os.path.join(vdir, "vision-rows.jsonl")):
        row = json.loads(line)
        b = row["benchmark"]
        d = per[b]
        blocks = R.blocks(row["questions"], row.get("prompt"))
        bl = [len(x[2]) for x in blocks]
        img_tok, mp = 0, []
        for p in row["images"]:
            with Image.open(os.path.join(vdir, p)) as im:
                W, H = im.size
            w, h = resize_dims(W, H, a.max_vis_tokens, 1.0)
            img_tok += (w // 32) * (h // 32)
            mp.append(W * H / 1e6)
        st = R.state_tokens(row["state"]) + img_tok + 2 * len(row["images"])
        nopt = [len(q["criteria"]) for q in row["questions"].values()]
        d["prefix"].append(st)
        d["img_tokens"].append(img_tok)
        d["n_img"].append(len(row["images"]))
        d["src_mp_max"].append(max(mp) if mp else 0)
        d["over_cap"].append(any(m > 1.6384 for m in mp))
        d["block_max"].append(max(bl))
        d["total"].append(st + sum(bl))
        d["nq"].append(len(bl))
        d["opt_max"].append(max(nopt))
        d["types"].extend(q["type"] for q in row["questions"].values())
        d["over26"].append(any(n > 26 for n in nopt))
        d["lat"].append(latency_ms(st - img_tok, sum(bl), len(bl), img_tok, a, any(n > 26 for n in nopt)))
    return summarize(per, a)


def summarize(per, a):
    out = {}
    for b, d in sorted(per.items()):
        n = len(d["prefix"])
        opt = collections.Counter(d["opt_max"])
        tot = np.array(d["total"])
        s = {
            "requests": n, "types": dict(collections.Counter(d["types"])),
            "questions_per_request": {"mean": float(np.mean(d["nq"])), "max": int(max(d["nq"]))},
            "options_max": int(max(d["opt_max"])), "options_hist": {str(k): v for k, v in sorted(opt.items())},
            "requests_over_26_options": int(sum(d["over26"])),
            "prefix_tokens": {"p50": pct(d["prefix"], 50), "p95": pct(d["prefix"], 95), "max": int(max(d["prefix"]))},
            "block_tokens_max": {"p50": pct(d["block_max"], 50), "p95": pct(d["block_max"], 95), "max": int(max(d["block_max"]))},
            "total_tokens": {"p50": pct(tot, 50), "p95": pct(tot, 95), "max": int(tot.max()), "mean": float(tot.mean())},
            "over_prefix_limit": int(sum(p > a.max_prefix for p in d["prefix"])),
            "over_block_limit": int(sum(x > a.max_block for x in d["block_max"])),
            "over_1536_state": int(sum(p > 1536 for p in d["prefix"])),
            "over_8192_total": int((tot > 8192).sum()),
            "est_latency_ms": {"p50": pct(d["lat"], 50), "mean": float(np.mean(d["lat"])), "p80": pct(d["lat"], 80),
                               "max": float(max(d["lat"])), "share_over_1000": float(np.mean(np.array(d["lat"]) > 1000))},
        }
        if "img_tokens" in d:
            s["images_per_request_max"] = int(max(d["n_img"]))
            s["image_tokens"] = {"p50": pct(d["img_tokens"], 50), "max": int(max(d["img_tokens"]))}
            s["source_megapixels_max"] = float(max(d["src_mp_max"]))
            s["requests_with_image_over_1.6MP"] = int(sum(d["over_cap"]))
        out[b] = s
    allat = [x for d in per.values() for x in d["lat"]]
    out["_all"] = {"requests": len(allat), "est_latency_ms": {"p50": pct(allat, 50), "mean": float(np.mean(allat)),
                                                             "p80": pct(allat, 80), "share_over_1000": float(np.mean(np.array(allat) > 1000))}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", help="text suite dir after `suite import` (suite-0.3/)")
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--rows", nargs="*", help="raw row files instead of --suite (edition read-time filter applied)")
    ap.add_argument("--vision", help="vision suite dir (work/vision)")
    ap.add_argument("--out", default=os.path.join(REPO, "di/suite_stats.json"))
    ap.add_argument("--max_vis_tokens", type=int, default=1600)
    ap.add_argument("--max_prefix", type=int, default=65536)
    ap.add_argument("--max_block", type=int, default=32768)
    ap.add_argument("--fixed_ms", type=float, default=70.0, help="per-request overhead (v1 measured 67 ms text_1q)")
    ap.add_argument("--tok_per_s", type=float, default=20000.0, help="prefill tokens/s of the shared-prefix forward")
    ap.add_argument("--per_block_ms", type=float, default=0.3)
    ap.add_argument("--vis_ms_per_tok", type=float, default=0.03, help="vision tower ms per merged image token")
    a = ap.parse_args()
    R = Renderer()
    res = {"assumptions": {k: getattr(a, k) for k in ("max_vis_tokens", "max_prefix", "max_block", "fixed_ms", "tok_per_s",
                                                      "per_block_ms", "vis_ms_per_tok")}}
    if a.suite or a.rows:
        res["text"] = text_stats(a.suite, a.edition, R, a, a.rows)
    if a.vision:
        res["vision"] = vision_stats(a.vision, R, a)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v.get("_all") for k, v in res.items() if isinstance(v, dict) and "_all" in v}, indent=1))


if __name__ == "__main__":
    main()
