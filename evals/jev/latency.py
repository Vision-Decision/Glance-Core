"""Jev latency from gen-host: one request at a time, wall time of the HTTP call (includes network + OpenRouter + Jev).

    python evals/jev/latency.py --root ~/jev --reps 60
Configs, interleaved round-robin so network drift hits all equally:
  box_1q / box_8q / box_32q : one COCO image (the gRefCOCO-val image with the most expressions; 612x612), Q box
                              questions about different expressions (same as Glance's latency protocol: Q box
                              questions sharing one image)
  pair_8x2                  : 8 expressions x (box + parallel noul) = 16 questions, the way the accuracy runs ask
  text_min                  : near-empty text request (state "x", one noul) -> network + OpenRouter overhead floor
Writes <root>/runs/latency/{samples.jsonl,summary.json}. The first `--warmup` rounds are discarded.
"""
import argparse, base64, json, os, statistics, sys, time
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import nat_q, NAT_STATE  # noqa: E402

URL = "https://openrouter.ai/api/alpha/decisions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/jev"))
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    reqs = [json.loads(l) for l in open(f"{a.root}/runs/grefcoco/requests.jsonl")]
    req = max(reqs, key=lambda r: len(r["items"]))
    exprs = [it["text"] for it in req["items"]][:32]
    img = base64.b64encode(open(req["images"]["photo"], "rb").read()).decode()
    asset = {"photo": dict(type="image", format="jpg", data_base64=img)}

    def box_body(q):
        return dict(model="typesafe/jev-1.14", state=NAT_STATE, assets=asset,
                    questions={f"q{k}_box": nat_q(e)["_box"] for k, e in enumerate(exprs[:q])})

    pair = dict(model="typesafe/jev-1.14", state=NAT_STATE, assets=asset,
                questions={f"q{k}{s}": v for k, e in enumerate(exprs[:8]) for s, v in nat_q(e).items()})
    cfgs = {"box_1q": box_body(1), "box_8q": box_body(8), "box_32q": box_body(32), "pair_8x2": pair,
            "text_min": dict(model="typesafe/jev-1.14", state="x",
                             questions={"q": dict(type="noul", instructions="Is the state the letter x?")})}
    payloads = {k: json.dumps(v) for k, v in cfgs.items()}
    os.makedirs(f"{a.root}/runs/latency", exist_ok=True)
    out = open(f"{a.root}/runs/latency/samples.jsonl", "a")
    sess = requests.Session()  # keep-alive, like a production client
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    samples = {k: [] for k in cfgs}
    for rep in range(a.reps + a.warmup):
        for k in (list(cfgs) if rep % 2 == 0 else list(cfgs)[::-1]):
            t0 = time.perf_counter()
            r = sess.post(URL, data=payloads[k], headers=hdr, timeout=120)
            ms = (time.perf_counter() - t0) * 1000
            ok = r.status_code == 200 and "answers" in r.text
            u = r.json().get("usage", {}) if ok else {}
            out.write(json.dumps(dict(cfg=k, rep=rep, warm=rep < a.warmup, status=r.status_code, ok=ok, ms=ms,
                                      n_q=len(cfgs[k]["questions"]), input_tokens=u.get("input_tokens"),
                                      cost=u.get("cost"), t=time.time())) + "\n"); out.flush()
            if ok and rep >= a.warmup:
                samples[k].append(ms)
    summ = {k: dict(n=len(v), ms_median=statistics.median(v), ms_mean=float(np.mean(v)), ms_p80=float(np.percentile(v, 80)),
                    ms_p90=float(np.percentile(v, 90)), ms_min=min(v), n_questions=len(cfgs[k]["questions"]))
            for k, v in samples.items() if v}
    summ["image_wh"] = req["items"][0]["W"], req["items"][0]["H"]
    summ["note"] = ("sequential requests from gen-host (GCP us-central1) over a keep-alive HTTPS session; wall time "
                    "includes network, OpenRouter routing and Jev. Glance's 67.0/66.7/70.2 ms (Q=1/8/32) are local GPU "
                    "forward time on an RTX PRO 6000 with no network.")
    json.dump(summ, open(f"{a.root}/runs/latency/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
