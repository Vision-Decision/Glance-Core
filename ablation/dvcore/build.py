"""Load base (bf16 / nf4 / fp8), inject LoRA into the LLM, attach box heads."""
import os, re, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict, set_peft_model_state_dict

from .backbone import SharedPrefixLM
from .heads import BoxHeads


def load_base(cfg, device="cuda"):
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if cfg.quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head", "in_proj_a", "in_proj_b", "mtp"])
        kw["device_map"] = {"": device}
    elif cfg.quant == "fp8":
        from transformers import FineGrainedFP8Config
        kw["quantization_config"] = FineGrainedFP8Config(modules_to_not_convert=["visual", "lm_head", "in_proj_a", "in_proj_b"])
        kw["device_map"] = {"": device}
    elif cfg.quant != "bf16":
        raise ValueError(cfg.quant)
    model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **kw)
    if cfg.quant == "bf16":
        model = model.to(device)
    proc = AutoProcessor.from_pretrained(cfg.model_id)
    return model, proc


def build(cfg, device="cuda", lora=True):
    hf, proc = load_base(cfg, device)
    hf.requires_grad_(False)
    hf.config.use_cache = False
    if lora and cfg.lora_r > 0:
        lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target, bias="none")
        inject_adapter_in_model(lc, hf)
        for n, p in hf.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
                p.data = p.data.float()  # fp32 master LoRA weights; forward runs under bf16 autocast
        # don't let peft cast every bf16 activation up to the fp32 adapter dtype (big copies); autocast handles the mm
        for m in hf.modules():
            if hasattr(m, "lora_A"):
                m.cast_input_dtype_enabled = False
    net = SharedPrefixLM(hf, cfg)
    heads = BoxHeads(net.hidden, cfg).to(device)  # fp32
    return net, heads, proc


def trainable_state(net, heads):
    return dict(lora=get_peft_model_state_dict(net.hf), heads=heads.state_dict())


def save_ckpt(path, net, heads, cfg, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(**trainable_state(net, heads), cfg=cfg.__dict__, extra=extra or {}), path)


def load_ckpt(path, net, heads):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if sd.get("lora"):
        res = set_peft_model_state_dict(net.hf, sd["lora"])
        bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
        assert not bad, f"unexpected LoRA keys: {bad[:5]}"
    heads.load_state_dict(sd["heads"])
    return sd
