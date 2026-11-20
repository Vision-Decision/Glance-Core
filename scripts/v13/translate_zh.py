"""v1.3 Chinese long positives: translate RefCOCO/+/g TRAIN expressions (the v1 natural pool; no eval image) into natural
Chinese with a self-hosted open LLM (vLLM, text only), keeping every attribute / location cue.

  python scripts/v13/translate_zh.py --pool pool_natural.jsonl --out zh_trans.jsonl --n 70000 [--model Qwen/Qwen3.8-27B-FP8]
Picks present questions with >= 3 words (the multi-attribute descriptions v1.2 learned to reject in Chinese), longest
first within a random sample. Output lines {key, image_id, en, zh}; key = "<image_id>|<english text>". Resumable.
"""
import argparse, json, os, random, re

SYS = ("You translate English referring expressions (a person pointing at one object in a photo) into natural, colloquial "
       "Simplified Chinese, the way a native speaker would say it. Keep every attribute, count, colour, action and location "
       "cue; add nothing; fix typos. Use everyday words (guy -> 男的/那个人, kid -> 小孩). Return ONLY JSON: "
       '{"zh": ["...", "..."]} with exactly one translation per input line, same order.')
CJK = re.compile(r"[一-鿿]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=70000); ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8"); ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items, seen = [], set()
    for l in open(a.pool):
        r = json.loads(l)
        for q in r["questions"]:
            t = q.get("text", "")
            if q.get("present") and q.get("box") and q.get("src", "").startswith("refcoco") and len(t.split()) >= 3:
                k = f"{r['image_id']}|{t}"
                if k not in seen:
                    seen.add(k); items.append((k, r["image_id"], t))
    rng.shuffle(items)
    items = items[: a.n]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["key"] for l in open(a.out)}
    items = [x for x in items if x[0] not in done]
    print(f"[zh] {len(items)} to translate ({len(done)} done)", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs)
    sp = SamplingParams(temperature=0.3, max_tokens=2000)
    f = open(a.out, "a")
    for c in range(0, len(items), a.batch * 512):
        chunk = items[c:c + a.batch * 512]
        bats = [chunk[i:i + a.batch] for i in range(0, len(chunk), a.batch)]
        convs = [[dict(role="system", content=SYS),
                  dict(role="user", content="\n".join(f"{j + 1}. {t}" for j, (_, _, t) in enumerate(b)))] for b in bats]
        outs = llm.chat(convs, sp, chat_template_kwargs=dict(enable_thinking=False), use_tqdm=False)
        n_ok = 0
        for b, o in zip(bats, outs):
            s = re.sub(r"^```(json)?|```$", "", o.outputs[0].text.strip(), flags=re.M).strip()
            try:
                zh = json.loads(s[s.index("{"):s.rindex("}") + 1])["zh"]
            except Exception:
                continue
            if len(zh) != len(b):
                continue
            for (k, iid, en), z in zip(b, zh):
                z = str(z).strip()
                if CJK.search(z) and not re.search(r"[A-Za-z]{3,}", z) and 2 <= len(z) <= 40:
                    f.write(json.dumps(dict(key=k, image_id=iid, en=en, zh=z), ensure_ascii=False) + "\n"); n_ok += 1
        f.flush()
        print(f"[zh] {c + len(chunk)}/{len(items)} kept {n_ok}", flush=True)
    print("[zh] DONE", flush=True)


if __name__ == "__main__":
    main()
