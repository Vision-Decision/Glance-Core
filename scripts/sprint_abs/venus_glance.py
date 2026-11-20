"""Glance model wrapper for the OFFICIAL VenusBench-GD evaluator (inclusionAI/UI-Venus, branch VenusBench-GD, eval.py).

Install: copy to <UI-Venus>/models/glance.py and add  "glance": ("models.glance", "GlanceModel")  to MODEL_MAP.
Env:
  GLANCE_CKPT     checkpoint path
  GLANCE_ANNS / GLANCE_IMAGES   the evaluator's --anns_dir / --images_dir (used to batch-precompute every item once)
  GLANCE_CACHE    json cache of the precomputed forward outputs (if it exists, the model is not even loaded)
  GLANCE_THR      presence threshold: result = "negative" iff p_present < GLANCE_THR (default 0.5)
  GLANCE_STACK    optional json {mu, sd, b}: p_present = sigmoid(((x - mu) / sd) . b[:-1] + b[-1]) with
                  x = [presence logit, logit(center top-1 prob), center entropy, IoU-head logit, log mean edge sd]
                  (recalibrator fitted on guisyn-fit only; never on VenusBench)
  GLANCE_GATED    1 (default): a refused item returns no point, exactly like a generative model that answers [-1,-1]
                  (positives refused => "wrong_format" => wrong). 0: always return the point (ungated click; NOT a
                  faithful mapping, only for comparison with the earlier own-script number).
  GLANCE_MAX_VIS / GLANCE_UPSCALE   (2048 / 2.0, as in the v1 eval)
The official loop calls inference(instruction, image_path) once per item and scores the returned point / result with
its own functions; the wrapper serves the identical Glance forward from a table precomputed in batches."""
import json, math, os, sys

DV = os.environ.get("DV_ROOT", "/content/dv")
sys.path.insert(0, DV)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class GlanceModel:
    def load_model(self, model_name_or_path=None):
        self.max_vis = int(os.environ.get("GLANCE_MAX_VIS", 2048)); self.up = float(os.environ.get("GLANCE_UPSCALE", 2.0))
        self.thr = float(os.environ.get("GLANCE_THR", 0.5)); self.gated = os.environ.get("GLANCE_GATED", "1") == "1"
        st = os.environ.get("GLANCE_STACK")
        self.stack = json.load(open(st)) if st else None
        self.table, self.net = {}, None
        cache = os.environ.get("GLANCE_CACHE")
        if cache and os.path.exists(cache):
            self.table = {tuple(json.loads(k)): v for k, v in json.load(open(cache)).items()}
            return
        self._build(model_name_or_path)
        if os.environ.get("GLANCE_ANNS"):
            self._precompute(os.environ["GLANCE_ANNS"], os.environ["GLANCE_IMAGES"])
            if cache:
                json.dump({json.dumps(list(k)): v for k, v in self.table.items()}, open(cache, "w"))

    def _build(self, ck=None):
        import torch
        from dv.config import Config
        from dv.model.build import build, load_ckpt
        ck = ck or os.environ["GLANCE_CKPT"]
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        self.cfg = Config().update(sd["cfg"])
        del sd
        self.net, self.heads, proc = build(self.cfg, "cuda")
        load_ckpt(ck, self.net, self.heads)
        self.net.cfg_proc = proc
        self.net.eval(); self.heads.eval()

    def set_generation_config(self, **kwargs):
        pass

    def _recs(self, items):
        from PIL import Image
        by = {}
        for path, ins in items:
            by.setdefault(path, []).append(ins)
        recs = []
        for path, ins in by.items():
            W, H = Image.open(path).size
            recs.append(dict(src="gui_eval", img=path, wh=[W, H], max_vis=self.max_vis, upscale=self.up, plen=0,
                             questions=[dict(type="box", text=t, present=True, box=None) for t in ins]))
        return recs

    def _run(self, items):
        if self.net is None:
            self._build()
        sys.path.insert(0, os.path.join(DV, "scripts", "sprint_abs"))
        from dump import predict_ext  # same forward as eval.predict_v1; rows carry (img, text) keys
        rows = predict_ext(self.net, self.heads, self._recs(items), "", self.cfg, prefix_budget=2048)
        for row in rows:
            self.table[(row["img"], row["text"])] = dict(p_logit=row["p_logit"], iou_logit=row["iou_logit"], box=row["box"],
                                                         wh=row["wh"], center_top1=row["center_top1"],
                                                         center_ent=row["center_ent"], edge_sd=row["edge_sd"])

    def _precompute(self, anns_dir, images_dir):
        items = []
        for fn in sorted(os.listdir(anns_dir)):
            if fn.endswith(".json"):
                for it in json.load(open(os.path.join(anns_dir, fn))):
                    items.append((os.path.join(images_dir, it["img_filename"]), it["instruction"]))
        self._run(list(dict.fromkeys(items)))

    def presence(self, o):
        if not self.stack:
            return 1 / (1 + math.exp(-o["p_logit"]))
        x = [o["p_logit"], _logit(o["center_top1"]), o["center_ent"], o["iou_logit"],
             math.log(sum(o["edge_sd"]) / len(o["edge_sd"]) + 1e-4)]
        s = self.stack
        z = sum((xi - m) / d * b for xi, m, d, b in zip(x, s["mu"], s["sd"], s["b"][:-1])) + s["b"][-1]
        return 1 / (1 + math.exp(-z))

    def inference(self, instruction, image_path):
        key = (image_path, instruction)
        if key not in self.table:
            self._run([key])
        o = self.table[key]
        W, H = o["wh"]; b = o["box"]
        p = self.presence(o)
        neg = p < self.thr
        point = [(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H]
        return dict(result="negative" if neg else "positive", format="x1y1x2y2",
                    bbox=None if (neg and self.gated) else [b[0] / W, b[1] / H, b[2] / W, b[3] / H],
                    point=None if (neg and self.gated) else point,
                    raw_response=json.dumps(dict(p_present=p, kappa=1 / (1 + math.exp(-o["iou_logit"])), box=b)))
