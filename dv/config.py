"""Single flat config. Defaults = v0 (Qwen3.5-4B-Base). Override with a JSON file and/or
`--set key=value` on the CLI. Nothing model-size specific is hard-coded elsewhere: layer types,
hidden size, conv kernel, head dims are read from the HF config at load time."""
import argparse, dataclasses, json
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- task ---
    task: str = "v0"                    # v0: box-only records (dv/data/dataset.py) | v1: mixed typed records (dv/data/mixed.py)
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
    w_iou_brier: float = 0.0            # sprint_abs: + Brier term on the IoU head (BCE + Brier, both proper)
    iou_head_on_sel: bool = False       # sprint_abs: also train the IoU head on the box decoded at the sampled patch
    # --- v1.2 multi-instance box readout (dv/model/multibox.py) ---
    multibox: bool = False              # add heat / match / count heads (keys heads.box.multi.*); single-box heads unchanged
    multi_k: int = 16                   # peak candidates per box question at inference
    multi_k_train: int = 8              # peak candidates per box question for the match (set) loss
    count_classes: int = 4              # P(n=1,2,3,>=4 | present); P(n=0) = 1 - presence gate
    heat_sigma_frac: float = 0.15       # heat target Gaussian sigma = frac * box size (patches), >= center_sigma_min
    w_heat: float = 1.0
    w_match: float = 0.5
    w_count: float = 0.5
    multi_match_thr: float = 0.5        # decode: keep candidates with P(match | present) >= thr
    lr_heat: float = 0.0                # >0: own LR for the heat map (box.multi.h_*), e.g. 1e-5 (v1.2b fix: heat loss diverged)
    clip_heat: float = 0.0              # >0: own grad-norm clip for the heat map
    heat_loss_norm: str = "pos"         # pos: CenterNet sum / #positives | pos_mean: also divide by #valid patches / 100
    multi_head: str = "heat"            # heat: peaks of the heat map + match/count heads | objectness: Glance-X per-patch
                                        # objectness (sigmoid focal on centre-region targets; candidates = top-k patches,
                                        # P(match) := objectness), ported from Vision-Decision/Glance-X (Apache-2.0) for the head-off
    obj_shrink: float = 0.5             # objectness: positive patches = central obj_shrink part of each box (+ nearest patch)
    multi_decode: str = "thr"           # thr | count (top-n by P(match), n = argmax of the count head; chosen on calib_mb)
    multi_backbone_grad: float = 1.0    # scale of the multi heads' gradient into the shared features / LoRA (joint stage)
    center_multi: bool = True           # False: centre-softmax loss only on questions with <= 1 referent box
    clip_multi_separately: bool = True  # clip heads.box.multi.* grads on their own (joint stage)
    lr_multi: float = 0.0               # lr of heads.box.multi.* (0: lr_heads)
    train_only: str = ""                # "multi": train only heads.box.multi.* (LoRA + all other heads frozen, backbone no-grad)
    # --- v1 typed primitives ---
    w_box: float = 1.0                  # weight of the whole box loss in a mixed micro-batch
    w_typed: float = 1.0                # weight of the mean typed (choice/noul/score) loss
    w_brier: float = 1.0                # choice / noul: CE + w_brier * Brier
    w_score_cramer: float = 1.0         # score: CE + w_score_cramer * squared Cramer over levels
    choice_smooth: float = 0.0          # v1.1: label smoothing on one-hot choice targets (train only)
    # --- robustness sprint (description shift) ---
    w_cons: float = 0.5                 # paraphrase-consistency loss across descriptions of the same target (0 = off);
                                        # default on since v1.2: only records whose box questions carry grp/anchor are affected
    cons_present_only: bool = True      # v1.2: pairs only where member and anchor are both present
    w_cons_pres: float = 1.0            #   presence: BCE(member logit, sigmoid(anchor logit) stop-grad)
    w_cons_center: float = 1.0          #   center: KL(anchor patch distribution (stop-grad) || member)
    w_cons_box: float = 1.0             #   box: L1(member decoded box, anchor decoded box (stop-grad))
    w_lm: float = 0.0                   # next-token LM loss on image captions (EN + ZH) through the plain HF forward
    lm_file: str = ""                   # jsonl {"img": rel path, "prompt": str, "caption": str}
    lm_bs: int = 4                      # caption samples per optimizer micro-step
    lm_max_vis: int = 448               # arch-exp: visual-token cap of the LM-loss images (GUI replay needs more)
    # --- Glance-Video ---
    video_max_vis: int = 96             # visual tokens per temporal group (2 frames) of a video prefix
    init_ckpt: str = ""                 # start from this checkpoint's LoRA + heads (not a resume: fresh optimizer/step)
    # --- data ---
    train_file: str = "data/prep/train.jsonl"
    image_root: str = "/content/data/coco/train2014"
    max_vis_tokens: int = 640           # cap visual tokens per image (1 token = 32x32 px)
    max_upscale: float = 2.0
    max_q_per_record: int = 48
    min_orig_frac: float = 0.0          # >0: at least this share of a record's sampled requests have orig=True (Glance-X mix)
    max_q_tokens: int = 40
    limit_records: int = 0              # >0: use only the first N records (overfit test)
    data_root: str = "/content/data"    # v1: record "img" paths are relative to this
    max_state_tokens: int = 1536        # v1: text state cap (prep drops longer states)
    max_q_tokens_typed: int = 384       # v1: typed block cap (prep drops longer blocks)
    prefix_budget: int = 12288          # v1: padded prefix tokens per micro-batch (B * max prefix len)
    q_budget: int = 192                 # v1: questions per micro-batch
    max_records: int = 32               # v1: records per micro-batch
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
    save_every_min: float = 0.0         # >0: also save ckpt_last.pt (+ opt_last.pt) every N minutes
    out_dir: str = "/content/out/v0"
    resume: str = ""

    @staticmethod
    def parse(argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", default="")
        ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
        ap.add_argument("rest", nargs="*")
        a, unknown = ap.parse_known_args(argv)
        c = Config()
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
        return Config(**new)

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=1)
