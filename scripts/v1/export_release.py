"""Export a v1 checkpoint for release (run on a GPU VM with the repo at /content/dv).

python scripts/v1/export_release.py --ckpt /content/out/v1_1/ckpt_final.pt --eval_json /content/out/v1_1/eval_results.json \
    --out /content/release/v1.1 [--merged]

Writes
  adapter/adapter_model.safetensors + adapter/adapter_config.json   LoRA (PEFT naming, bf16)
  heads.safetensors                                                 box + typed heads (fp32)
  dv_config.json                                                    model/prompt config, label token ids, temperatures
  merged-bf16/  (--merged)                                          base + LoRA merged, HF format (+ heads.safetensors)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from safetensors.torch import save_file

from dv.config import Config
from dv.model.build import build, load_ckpt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--eval_json", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--merged", action="store_true")
a = ap.parse_args()
os.makedirs(f"{a.out}/adapter", exist_ok=True)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = Config().update(sd["cfg"])
# ---- adapter (PEFT naming) ----
lora = {k: v.to(torch.bfloat16).contiguous() for k, v in sd["lora"].items()}
save_file(lora, f"{a.out}/adapter/adapter_model.safetensors")
json.dump(dict(peft_type="LORA", base_model_name_or_path=cfg.model_id, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
               lora_dropout=0.0, target_modules=cfg.lora_target, bias="none", task_type=None,
               inference_mode=True), open(f"{a.out}/adapter/adapter_config.json", "w"), indent=1)
heads = {k: v.float().contiguous() for k, v in sd["heads"].items()}
save_file(heads, f"{a.out}/heads.safetensors")
temps = {}
if a.eval_json and os.path.exists(a.eval_json):
    ev = json.load(open(a.eval_json))
    temps = ev.get("temps_choice_hard") or ev.get("temps") or {}
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(cfg.model_id)
from dv.data.mixed import ChatTok
tok = ChatTok(proc, cfg)
json.dump(dict(dv_version="v1.1", base_model=cfg.model_id, train_step=sd["extra"].get("step"), config=cfg.__dict__,
               temperatures=temps, label_token_groups=tok.label_token_groups(),
               prompt_layout="see dv/data/mixed.py (chat-style prefix; one block per question)"),
          open(f"{a.out}/dv_config.json", "w"), indent=1)
print("adapter", sum(v.numel() for v in lora.values()) / 1e6, "M; heads", sum(v.numel() for v in heads.values()) / 1e6, "M")
# ---- merged bf16 ----
if a.merged:
    net, hd, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, hd)
    hf = net.hf
    n = 0
    for name, m in hf.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "merge"):
            m.merge(); n += 1
    state = {}
    for k, v in hf.state_dict().items():
        if "lora_" in k:
            continue
        state[k.replace(".base_layer.", ".")] = v.to(torch.bfloat16)
    # reload a clean model skeleton and load the merged weights, then save in HF format
    from transformers import AutoModelForImageTextToText
    clean = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    missing, unexpected = clean.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    assert all("lm_head" in m or "mtp" in m for m in missing), missing[:5]
    clean.save_pretrained(f"{a.out}/merged-bf16", safe_serialization=True, max_shard_size="5GB")
    proc.save_pretrained(f"{a.out}/merged-bf16")
    save_file(heads, f"{a.out}/merged-bf16/heads.safetensors")
    print("merged", n, "LoRA modules ->", f"{a.out}/merged-bf16")
