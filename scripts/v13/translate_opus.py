"""Chinese translations of RefCOCO-train expressions with Claude Opus 5.5 through the subscription CLI (gen-host):
`claude -p --model claude-opus-5-5`, CLAUDE_CONFIG_DIR=~/.claude-bench, token from ~/.claude-bench/bench.env (read into the
child environment only; never printed), ANTHROPIC_API_KEY removed. Batches of --batch items per call, --workers concurrent
calls (~3), every batch cached in --out (resumable).

  python3 scripts/v13/translate_opus.py --items items.jsonl --out zh_opus.jsonl [--limit 200 for the pilot]
items.jsonl lines: {"key", "image_id", "en"}.  Output lines: {"key", "image_id", "en", "zh", "model"}.
"""
import argparse, json, os, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor

PROMPT = """Translate each English referring expression below (someone pointing at ONE object in a photo) into natural,
colloquial Simplified Chinese, as a native speaker would say it. Keep every attribute, count, colour, action and location
cue; add nothing; silently fix typos (e.g. "tsp" -> T-shirt only if obvious from context, otherwise keep the meaning literal).
Use everyday words (guy -> 男的/那个人, kid -> 小孩). Output ONLY a JSON object {"zh": [...]} with exactly one string per
input line, in the same order, nothing else.

"""
CJK = re.compile(r"[一-鿿]")


def env():
    e = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    e["CLAUDE_CONFIG_DIR"] = os.path.expanduser("~/.claude-bench")
    for l in open(os.path.expanduser("~/.claude-bench/bench.env")):
        if "=" in l and not l.startswith("#"):
            k, v = l.strip().split("=", 1)
            e[k] = v.strip().strip('"')
    return e


def call(batch, e, timeout=600):
    text = PROMPT + "\n".join(f"{i + 1}. {x['en']}" for i, x in enumerate(batch))
    r = subprocess.run(["claude", "-p", "--model", "claude-opus-5-5", "--output-format", "text"], input=text,
                       capture_output=True, text=True, env=e, timeout=timeout)
    s = r.stdout.strip()
    s = s[s.index("{"):s.rindex("}") + 1]
    zh = json.loads(s)["zh"]
    if len(zh) != len(batch):
        raise ValueError(f"{len(zh)} != {len(batch)}")
    return zh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=100); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    done = {json.loads(l)["key"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [x for x in items if x["key"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    bats = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    print(f"[opus] {len(todo)} items, {len(bats)} calls ({len(done)} cached)", flush=True)
    e = env(); lock = threading.Lock(); f = open(a.out, "a"); stat = dict(ok=0, fail=0, kept=0); t0 = time.time()

    def work(b):
        for att in range(3):
            try:
                zh = call(b, e)
                with lock:
                    for x, z in zip(b, zh):
                        z = str(z).strip()
                        if CJK.search(z) and not re.search(r"[A-Za-z]{3,}", z) and 2 <= len(z) <= 40:
                            f.write(json.dumps(dict(x, zh=z, model="claude-opus-5-5"), ensure_ascii=False) + "\n"); stat["kept"] += 1
                    f.flush(); stat["ok"] += 1
                return
            except Exception as ex:
                time.sleep(30 * (att + 1))
                err = type(ex).__name__
        with lock:
            stat["fail"] += 1
            print("[opus] batch failed", err, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(work, bats)):
            if i % 10 == 0:
                print(f"[opus] {i + 1}/{len(bats)} {stat} {time.time() - t0:.0f}s", flush=True)
    print("[opus] DONE", stat, flush=True)


if __name__ == "__main__":
    main()
