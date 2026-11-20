"""Fair latency (experiment 5a): generative Qwen3.5-4B served by vLLM (async engine, in-process) with automatic
prefix caching.  Runs in its own venv:  /content/vllm_env/bin/python ablation/latency_vllm.py --out OUT.json

Workload identical to ablation/latency_v2.py: image i = i-th record of eval_refcoco_val.jsonl resized to <= 640
visual tokens with sides multiples of 32 (the Qwen processor keeps such images unchanged; prompt_tokens_mean
confirms the visual-token count), N questions = the record's expressions cycled to N.
Prompt = baseline.py's grounding prompt (chat template, thinking off); greedy; max_tokens 24 (the box string
"[x1, y1, x2, y2]" is ~17-20 tokens; AR-digits needs <= 20).
Every call submits all ipb*N requests at once to the async engine (vLLM batches them, shares cached image-prefix
blocks between requests of the same image once computed); the prefix cache is reset before each call so nothing
is reused across calls. Per call: total time until the last answer, time-to-first-answer (first finished request),
generated tokens, cached prompt tokens per request (evidence that prefix caching is active).
Modes: bs1_N{1,2,4,8,16,32}, bs8_N{1,8}; --no_prefix_caching reruns as an ablation.
Submission strategies (--strategies): "simul" submits all requests at once (requests scheduled in the same engine
step cannot reuse each other's not-yet-computed prefix blocks, so sharing is partial); "staged" = best case for
prefix caching: per image, the first question is submitted alone and the remaining N-1 only once its first token
has been produced (its image+prompt prefix blocks are then cached), images of a batch proceed concurrently.
"""
import argparse, asyncio, json, math, time, uuid
import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixed_q import mixed_questions, typed_body  # noqa: E402

PROMPT = ('Locate the object described as "{expr}" in the image. Output its bounding box as [x1, y1, x2, y2] '
          'with coordinates normalized to 0-1000. If no such object is in the image, answer exactly "none".')
MODES = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (8, 1), (8, 8)]
# --mixed: Q mixed questions per image (see ablation/mixed_q.py); results keyed bs1_N{Q}


def resize_dims(W, H, max_tokens, max_upscale, factor=32):   # = ablation/dvcore/dataset.resize_dims
    s = min(math.sqrt(max_tokens * factor * factor / (W * H)), max_upscale)
    while True:
        w = max(factor, int(round(W * s / factor)) * factor)
        h = max(factor, int(round(H * s / factor)) * factor)
        if (w // factor) * (h // factor) <= max_tokens:
            return w, h
        s *= 0.98


def n_calls_for(ipb, nq, n_images):
    n = n_images // ipb
    return n if nq <= 8 else max(20, n // 2)


async def run(a):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except Exception:
        from vllm import AsyncLLMEngine as Engine
    from transformers import AutoProcessor
    import vllm, torch
    proc = AutoProcessor.from_pretrained(a.model)
    eng = Engine.from_engine_args(AsyncEngineArgs(model=a.model, enable_prefix_caching=not a.no_prefix_caching,
                                                  max_model_len=4096, limit_mm_per_prompt={"image": 1},
                                                  gpu_memory_utilization=0.85, seed=0))
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    recs = [json.loads(l) for l in open(a.prep)][: 2 * a.n_images]
    imgs, qs = [], []
    for r in recs:
        img = Image.open(f"{a.image_root}/{r['file']}").convert("RGB")
        w, h = resize_dims(*img.size, 640, 2.0)
        imgs.append(img.resize((w, h), Image.BICUBIC))
        qs.append([q["text"] for q in r["questions"]])

    def req(i, e):
        text_q = PROMPT.format(expr=e) if isinstance(e, str) else (PROMPT.format(expr=e["text"]) if e["type"] == "box" else typed_body(e))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return {"prompt": text, "multi_modal_data": {"image": imgs[i]}}

    async def one(r, t0, ev=None):
        final = None
        async for out in eng.generate(r, sp, request_id=uuid.uuid4().hex):
            final = out
            if ev is not None and not ev.is_set():
                ev.set()
        if ev is not None:
            ev.set()
        return time.perf_counter() - t0, final

    async def staged(rs, t0):
        ev = asyncio.Event()
        t_first = asyncio.ensure_future(one(rs[0], t0, ev))
        await ev.wait()
        rest = await asyncio.gather(*[one(r, t0) for r in rs[1:]])
        return [await t_first] + list(rest)

    res = dict(mixed=a.mixed, vllm=vllm.__version__, torch=torch.__version__, model=a.model, prefix_caching=not a.no_prefix_caching,
               max_tokens=a.max_tokens, n_images=a.n_images, gpu=torch.cuda.get_device_name(), modes={})
    for strat, (ipb, nq) in [(s_, m) for s_ in a.strategies.split(",") for m in MODES]:
        if strat == "staged" and nq == 1:
            continue
        T, F, ntok, ptok, ctok, ex = [], [], [], [], [], []
        for c in range(n_calls_for(ipb, nq, a.n_images) + a.warmup):
            idx = [(c * ipb + j) % len(imgs) for j in range(ipb)]
            if a.mixed:
                reqs = [req(i, q) for i in idx for q in mixed_questions(qs[i], nq)]
            else:
                reqs = [req(i, qs[i][j % len(qs[i])]) for i in idx for j in range(nq)]
            await eng.reset_prefix_cache()
            t0 = time.perf_counter()
            if strat == "staged":
                grp = await asyncio.gather(*[staged(reqs[g * nq:(g + 1) * nq], t0) for g in range(ipb)])
                outs = [o for g in grp for o in g]
            else:
                outs = await asyncio.gather(*[one(r, t0) for r in reqs])
            t1 = time.perf_counter()
            if c >= a.warmup:
                T.append((t1 - t0) * 1000); F.append(min(o[0] for o in outs) * 1000)
                for _, o in outs:
                    ntok.append(len(o.outputs[0].token_ids)); ptok.append(len(o.prompt_token_ids))
                    ctok.append(getattr(o, "num_cached_tokens", None) or 0)
                if len(ex) < 6:
                    ex.append(outs[0][1].outputs[0].text)
        t, f = np.array(T), np.array(F)
        name = f"bs{ipb}_N{nq}" + ("" if strat == "simul" else "_staged")
        res["modes"][name] = dict(ms_median=float(np.median(t)), ms_p90=float(np.percentile(t, 90)), ms_mean=float(t.mean()),
                                  q_per_s=float(ipb * nq * len(t) / (t.sum() / 1000)), ttfa_ms_median=float(np.median(f)),
                                  ttfa_ms_p90=float(np.percentile(f, 90)), n_calls=len(t), questions_per_call=ipb * nq,
                                  gen_tokens_mean=float(np.mean(ntok)), prompt_tokens_mean=float(np.mean(ptok)),
                                  cached_prompt_tokens_mean=float(np.mean(ctok)), examples=ex)
        print(name, json.dumps({k: v for k, v in res["modes"][name].items() if k != "examples"}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res, open(a.out, "w"), indent=1)
    try:
        eng.shutdown()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=24)
    ap.add_argument("--no_prefix_caching", action="store_true")
    ap.add_argument("--strategies", default="simul,staged")
    ap.add_argument("--mixed", action="store_true", help="mixed typed workload (ablation/mixed_q.py): box/yes-no/choice/score")
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
