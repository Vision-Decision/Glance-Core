"""Shrink a checkpoint for download: LoRA tensors -> bf16 (heads stay fp32). python scripts/export_ckpt.py IN OUT"""
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd["lora"] = {k: v.to(torch.bfloat16) for k, v in sd["lora"].items()}
sd["extra"] = {k: v for k, v in sd["extra"].items() if k != "opt"}
torch.save(sd, sys.argv[2]); print("saved", sys.argv[2])
