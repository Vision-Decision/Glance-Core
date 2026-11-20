"""Resumable OpenRouter client for TypeSafe Jev decisions (POST /api/alpha/decisions).

    python evals/jev/client.py --requests runs/grefcoco/requests.jsonl --out runs/grefcoco [--workers 8] [--limit 20]

Input: JSONL, one request per line:
    {"id": str, "state": str|obj, "questions": {key: {type, instructions, criteria?}},
     "images": {ref: "/abs/path.png"}, ...extra fields (ignored here, used by the scorer)}
Output under --out:
    raw/<id>.json      one per successful request: wall latency of the successful attempt (measured here), attempts,
                       image prep (original/sent size, scale), and the full response JSON
    fail/<id>.json     final failures (never silently dropped; retried on the next run unless --skip_failed)
    attempts.jsonl     every HTTP attempt: id, attempt, status, ms
Images: TypeSafe accepts 280..1568 px per side and <= 16 MiB. Larger images are downscaled (aspect kept) and
smaller ones upscaled; Jev's box is in 0..1 fractions of the sent image, so the scorer maps it back with the
original size (the scale is recorded anyway). Images inside the limits are sent byte-for-byte.
Transient errors (429, 5xx, timeouts, connection errors) are retried with exponential backoff; a 429 pauses all
workers. The API key is read from OPENROUTER_API_KEY and never written anywhere.
"""
import argparse, base64, io, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"
MIN_SIDE, MAX_SIDE, MAX_BYTES = 280, 1568, 16 * 1024 * 1024
TRANSIENT = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def safe(i):
    return str(i).replace("/", "__")


def prep_image(path):
    im = Image.open(path)
    W, H = im.size
    fmt = (im.format or "").upper()
    s = 1.0
    if max(W, H) > MAX_SIDE:
        s = MAX_SIDE / max(W, H)
    if min(W, H) * s < MIN_SIDE:
        s = MIN_SIDE / min(W, H)  # may violate MAX_SIDE for extreme aspect ratios; the API error is then recorded
    raw = open(path, "rb").read()
    if s == 1.0 and fmt in ("JPEG", "PNG") and len(raw) <= MAX_BYTES:
        data, out_fmt, sw, sh = raw, "jpg" if fmt == "JPEG" else "png", W, H
    else:
        sw, sh = max(1, round(W * s)), max(1, round(H * s))
        im2 = im.convert("RGB").resize((sw, sh), Image.LANCZOS)
        buf = io.BytesIO()
        if fmt == "PNG":
            im2.save(buf, "PNG"); out_fmt = "png"
        else:
            im2.save(buf, "JPEG", quality=95); out_fmt = "jpg"
        data = buf.getvalue()
    return dict(format=out_fmt, data_base64=base64.b64encode(data).decode()), \
        dict(orig_wh=[W, H], sent_wh=[sw, sh], scale=s, format=out_fmt, bytes=len(data))


class Pacer:
    """Global pause after a 429 so all workers back off together."""
    def __init__(self):
        self.until, self.lock = 0.0, threading.Lock()

    def wait(self):
        d = self.until - time.time()
        if d > 0:
            time.sleep(d)

    def pause(self, sec):
        with self.lock:
            self.until = max(self.until, time.time() + sec)


def run_one(req, out, key, pacer, a, alog, alock):
    rid = safe(req["id"])
    assets, imeta = {}, {}
    for ref, path in (req.get("images") or {}).items():
        asset, m = prep_image(path)
        assets[ref] = dict(type="image", **asset)
        imeta[ref] = m
    body = dict(model=MODEL, state=req["state"], questions=req["questions"])
    if assets:
        body["assets"] = assets
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if a.danger_header:
        headers["TypeSafe-Danger-Zone-Images"] = "true"
    payload = json.dumps(body)
    last = None
    for att in range(1, a.max_attempts + 1):
        pacer.wait()
        t0 = time.time()
        status, txt = None, ""
        try:
            r = requests.post(URL, data=payload, headers=headers, timeout=a.timeout)
            status, txt = r.status_code, r.text
        except requests.RequestException as e:
            txt = f"{type(e).__name__}: {str(e)[:300]}"
        ms = (time.time() - t0) * 1000
        with alock:
            alog.write(json.dumps(dict(id=req["id"], attempt=att, status=status, ms=round(ms, 1), t=round(t0, 3))) + "\n")
            alog.flush()
        if status == 200:
            try:
                resp = json.loads(txt)
            except ValueError:
                resp = None
            if resp is not None and "answers" in resp:
                rec = dict(id=req["id"], ok=True, latency_ms=ms, attempts=att, t=t0, n_questions=len(req["questions"]),
                           img=imeta, resp=resp)
                tmp = f"{out}/raw/{rid}.json.tmp"
                json.dump(rec, open(tmp, "w")); os.replace(tmp, f"{out}/raw/{rid}.json")
                if os.path.exists(f"{out}/fail/{rid}.json"):
                    os.remove(f"{out}/fail/{rid}.json")
                return "ok", resp.get("usage", {}).get("cost", 0.0), ms
            last = dict(status=status, body=txt[:2000], note="200 without answers")
        else:
            last = dict(status=status, body=txt[:2000])
            if status is not None and status not in TRANSIENT:
                break
        if status == 429:
            ra = None
            try:
                ra = float(r.headers.get("Retry-After", ""))
            except (ValueError, NameError):
                pass
            pacer.pause(ra if ra else min(60, 5 * 2 ** (att - 1)) + random.random())
        time.sleep(min(60, 2 ** att) * (0.5 + random.random()))
    json.dump(dict(id=req["id"], attempts=att, img=imeta, t=time.time(), **last), open(f"{out}/fail/{rid}.json", "w"))
    return "fail", 0.0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma list: only these ids")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--max_attempts", type=int, default=7)
    ap.add_argument("--skip_failed", action="store_true")
    ap.add_argument("--danger_header", action="store_true", help="send TypeSafe-Danger-Zone-Images: true")
    ap.add_argument("--budget_usd", type=float, default=0, help="stop submitting once this much usage.cost is spent")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    os.makedirs(f"{a.out}/raw", exist_ok=True); os.makedirs(f"{a.out}/fail", exist_ok=True)
    reqs = [json.loads(l) for l in open(a.requests)]
    if a.ids:
        want = set(a.ids.split(",")); reqs = [r for r in reqs if r["id"] in want]
    if a.limit:
        reqs = reqs[: a.limit]
    done = {f[:-5] for f in os.listdir(f"{a.out}/raw") if f.endswith(".json")}
    failed = {f[:-5] for f in os.listdir(f"{a.out}/fail") if f.endswith(".json")}
    todo = [r for r in reqs if safe(r["id"]) not in done and not (a.skip_failed and safe(r["id"]) in failed)]
    print(f"[client] {len(reqs)} requests, {len(reqs) - len(todo)} cached, {len(todo)} to run, workers={a.workers}",
          flush=True)
    pacer, alock = Pacer(), threading.Lock()
    alog = open(f"{a.out}/attempts.jsonl", "a")
    n_ok = n_fail = 0; cost = 0.0; t0 = time.time(); stop = threading.Event()

    def task(r):
        if stop.is_set():
            return "skipped", 0.0, None
        return run_one(r, a.out, key, pacer, a, alog, alock)

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(task, r) for r in todo]
        for k, f in enumerate(as_completed(futs), 1):
            st, c, ms = f.result()
            n_ok += st == "ok"; n_fail += st == "fail"; cost += c or 0.0
            if a.budget_usd and cost > a.budget_usd and not stop.is_set():
                print(f"[client] budget {a.budget_usd} reached (cost {cost:.4f}); stopping", flush=True); stop.set()
            if k % 50 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"[client] {k}/{len(futs)} ok={n_ok} fail={n_fail} cost=${cost:.4f} {k / el:.2f} req/s", flush=True)
    print(json.dumps(dict(done=n_ok, failed=n_fail, cost_usd=cost, wall_s=time.time() - t0)), flush=True)


if __name__ == "__main__":
    main()
