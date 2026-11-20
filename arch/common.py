"""Shared helpers for the architecture experiments (arch/)."""
import torch


def lora_modules(net):
    return [m for m in net.hf.modules() if hasattr(m, "lora_A") and hasattr(m, "scaling")]


def set_lora_scale(net, alpha, base=None):
    """Scale every LoRA delta by alpha (W = W0 + alpha * s * B A). Records the original scaling on first call so repeated
    calls are absolute, not cumulative. alpha=0 is the base model's LLM with the Glance heads attached."""
    n = 0
    for m in lora_modules(net):
        if not hasattr(m, "_scaling0"):
            m._scaling0 = dict(m.scaling)
        for k, v in m._scaling0.items():
            m.scaling[k] = v * alpha
        n += 1
    assert n > 0, "no LoRA modules found"
    return n
