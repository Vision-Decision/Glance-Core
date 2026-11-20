"""[ablation snapshot of dv/config.py; parse/update made subclass-friendly]
Single flat config. Defaults = v0 (Qwen3.5-4B-Base). Override with a JSON file and/or
`--set key=value` on the CLI. Nothing model-size specific is hard-coded elsewhere: layer types,
hidden size, conv kernel, head dims are read from the HF config at load time."""
import argparse, dataclasses, json
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- model ---
    model_id: str = "Qwen/Qwen3.5-4B"   # aligned model (coordinator decision; Base also works)
    quant: str = "bf16"                 # bf16 | nf4 | fp8   (base weights; LoRA/heads stay bf16/fp32)
    attn_mask: str = "block_bidir"      # block_bidir | causal  (question block internal mask on full-attn layers)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    # regex on module names; only LLM linears (vision tower stays frozen, no LoRA)
    lora_target: str = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
    grad_ckpt: bool = True
    # --- heads ---
    head_dim: int = 512
    edge_bins: int = 64
    edge_lo: float = -0.1               # edge offset range (fraction of image side) relative to chosen patch center
    edge_hi: float = 1.0
    edge_sigma_bins: float = 1.0        # Gaussian smoothing of the ordinal target
    center_target: str = "box"         # box: area-weighted over all patches covered by GT box (GUI-Actor) | gauss
    center_sigma_frac: float = 0.15     # Gaussian center target sigma = frac * box size (in patches), clamped
    center_sigma_min: float = 0.5
    center_sigma_max: float = 2.0
    w_presence: float = 1.0
    w_center: float = 1.0
    w_cramer: float = 0.1
    w_giou: float = 2.0
    w_iou_head: float = 0.5
    # --- data ---
    train_file: str = "data/prep/train.jsonl"
    image_root: str = "/content/data/coco/train2014"
    max_vis_tokens: int = 640           # cap visual tokens per image (1 token = 32x32 px)
    max_upscale: float = 2.0
    max_q_per_record: int = 48
    max_q_tokens: int = 40
    limit_records: int = 0              # >0: use only the first N records (overfit test)
    # --- train ---
    images_per_step: int = 4            # images (records) per forward
    grad_accum: int = 2
    epochs: float = 1.0
    max_steps: int = 0                  # >0 overrides epochs
    lr: float = 2e-4                    # LoRA
    lr_heads: float = 5e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.05
    optim: str = "adamw"                # adamw | adamw8bit
    clip: float = 1.0
    num_workers: int = 8
    seed: int = 0
    log_every: int = 10
    save_every: int = 1000
    out_dir: str = "/content/out/v0"
    resume: str = ""

    @classmethod
    def parse(cls, argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", default="")
        ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
        ap.add_argument("rest", nargs="*")
        a, unknown = ap.parse_known_args(argv)
        c = cls()
        if a.cfg:
            c = c.update(json.load(open(a.cfg)))
        kv = {}
        for s in a.set + [u.lstrip("-") for u in unknown if "=" in u]:
            k, v = s.split("=", 1)
            kv[k] = v
        c = c.update(kv, from_str=True)
        return c, a

    def update(self, d, from_str=False):
        fields = {f.name: f for f in dataclasses.fields(self)}
        new = dataclasses.asdict(self)
        for k, v in d.items():
            if k not in fields:
                raise KeyError(f"unknown config key {k}")
            if from_str:
                t = type(new[k])
                v = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
            new[k] = v
        return type(self)(**new)

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=1)
