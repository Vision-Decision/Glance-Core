"""Back-translate the Qwen Chinese translations to English with the same model (vLLM), for the round-trip filter
(filter_bt.py).  python scripts/v13/backtrans.py --zh zh_trans.jsonl --out zh_bt.jsonl [--skip_keys zh_opus.jsonl]"""
import argparse, json, os, re

SYS = ("Translate each Chinese phrase below (someone pointing at one object in a photo) into plain English, literally: keep "
       "every attribute, colour, count, action and location, add nothing. Return ONLY JSON {\"en\": [...]} with one string "
       "per input line, same order.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", required=True); ap.add_argument("--out", required=True); ap.add_argument("--skip_keys", default="")
    ap.add_argument("--batch", type=int, default=25); ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--max_model_len", type=int, default=4096); ap.add_argument("--gpu_mem", type=float, default=0.9)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    a = ap.parse_args()
    skip = {json.loads(l)["key"] for l in open(a.skip_keys)} if a.skip_keys and os.path.exists(a.skip_keys) else set()
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    items = [z for z in map(json.loads, open(a.zh)) if z["key"] not in skip and z["key"] not in done]
    print(f"[bt] {len(items)} to back-translate", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs)
    sp = SamplingParams(temperature=0.0, max_tokens=2000)
    f = open(a.out, "a")
    for c in range(0, len(items), a.batch * 512):
        chunk = items[c:c + a.batch * 512]
        bats = [chunk[i:i + a.batch] for i in range(0, len(chunk), a.batch)]
        convs = [[dict(role="system", content=SYS), dict(role="user", content="\n".join(f"{j + 1}. {z['zh']}" for j, z in enumerate(b)))]
                 for b in bats]
        outs = llm.chat(convs, sp, chat_template_kwargs=dict(enable_thinking=False), use_tqdm=False)
        for b, o in zip(bats, outs):
            s = re.sub(r"^```(json)?|```$", "", o.outputs[0].text.strip(), flags=re.M).strip()
            try:
                en = json.loads(s[s.index("{"):s.rindex("}") + 1])["en"]
            except Exception:
                continue
            if len(en) != len(b):
                continue
            for z, e in zip(b, en):
                f.write(json.dumps(dict(z, bt=str(e).strip()), ensure_ascii=False) + "\n")
        f.flush()
        print(f"[bt] {c + len(chunk)}/{len(items)}", flush=True)
    print("[bt] DONE", flush=True)


if __name__ == "__main__":
    main()
