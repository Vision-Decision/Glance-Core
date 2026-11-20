"""Minimal single-threaded `POST /v1/systemone` server around di.engine.GlanceEngine (one request at a time, as the
board measures latency). Lets the maintainers (or the kit's `http` engine) drive the model without importing it.

    PYTHONPATH=. python -m di.server --port 8000            # weights: Vision-Decision/Glance-4B-preview
    python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8000 --option model=glance-v1 ...

Body: {"model", "state", "questions", optional "images": [data URL | base64 | path], optional "prompt": {...}}.
Over-limit requests -> HTTP 422 whose text carries the kit's capacity markers ("maximum context length",
"options per choice"), which the `http` engine records as `unsupported`. Other failures -> HTTP 500.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from di.engine import GlanceEngine, Unsupported

ENGINE = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            return self._send(200, {"status": "ok", "model": ENGINE.model_name})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            resp, raw = ENGINE(req.get("state"), req["questions"], images=req.get("images") or (), prompt=req.get("prompt"))
            resp["evaluation_trace"] = raw
            self._send(200, resp)
        except Unsupported as e:
            msg = str(e)
            if "option" in msg and "options per choice" not in msg:
                msg += " (options per choice)"
            self._send(422, {"error": msg})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="Vision-Decision/Glance-4B-preview", help="HF release repo or local release dir")
    ap.add_argument("--ckpt", default=None, help="training checkpoint .pt (overrides --model)")
    ap.add_argument("--temps", default=None, help="eval_results.json (overrides the release temperatures)")
    ap.add_argument("--option", action="append", default=[], help="extra GlanceEngine option key=value")
    a = ap.parse_args()
    opts = dict(o.split("=", 1) for o in a.option)
    ENGINE = GlanceEngine(model=a.model, ckpt=a.ckpt, temps=a.temps, **opts)
    ENGINE.warmup()
    print(json.dumps({"event": "ready", "port": a.port, "provenance": ENGINE.provenance}), flush=True)
    HTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
