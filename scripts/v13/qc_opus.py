"""Claude Opus 5.5 quality check of a random sample of the kept (round-trip-filtered) Qwen translations: for each English
referring expression and its Chinese translation, is the translation faithful enough to point at the same object (head noun,
every attribute, count and location kept, nothing added)?  Subscription CLI as in translate_opus.py; 100 pairs per call.

  python3 scripts/v13/qc_opus.py --pairs zh_filtered.jsonl --n 500 --out qc_opus.jsonl
"""
import argparse, json, random, subprocess
from concurrent.futures import ThreadPoolExecutor
from translate_opus import env

PROMPT = """You are checking machine translations of English referring expressions (someone pointing at ONE object in a
photo) into Chinese. For each numbered pair, answer "ok" if the Chinese would make a native speaker point at the same object
(same head noun or an acceptable equivalent, every attribute / colour / count / action / location kept, nothing important
added), otherwise "bad" with a 3-8 word reason. Return ONLY JSON {"r": [["ok", ""], ["bad", "drops colour"], ...]} with one
entry per pair, same order.

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--n", type=int, default=500); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rows = [r for r in map(json.loads, open(a.pairs)) if r.get("model") != "claude-opus-5-5"]
    sample = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    bats = [sample[i:i + 100] for i in range(0, len(sample), 100)]
    e = env()

    def call(b):
        text = PROMPT + "\n".join(f"{i + 1}. EN: {x['en']} | ZH: {x['zh']}" for i, x in enumerate(b))
        out = subprocess.run(["claude", "-p", "--model", "claude-opus-5-5", "--output-format", "text"], input=text,
                             capture_output=True, text=True, env=e, timeout=900).stdout
        r = json.loads(out[out.index("{"):out.rindex("}") + 1])["r"]
        assert len(r) == len(b)
        return [dict(x, qc=v[0], why=v[1] if len(v) > 1 else "") for x, v in zip(b, r)]

    with ThreadPoolExecutor(3) as ex:
        res = [y for ys in ex.map(call, bats) for y in ys]
    with open(a.out, "w") as f:
        for y in res:
            f.write(json.dumps(y, ensure_ascii=False) + "\n")
    bad = [y for y in res if y["qc"] != "ok"]
    print(json.dumps(dict(n=len(res), bad=len(bad), bad_rate=round(len(bad) / max(1, len(res)), 4))))
    for y in bad[:15]:
        print(y["en"], "=>", y["zh"], "|", y["why"])


if __name__ == "__main__":
    main()
