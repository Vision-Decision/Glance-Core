"""Shrink a checkpoint for download: LoRA (+ new-token embeddings) to bf16; heads stay fp32.  python -m ablation.export_bf16 in.pt out.pt"""
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd["lora"] = {k: v.to(torch.bfloat16) for k, v in sd["lora"].items()}
if sd.get("new_emb") is not None:
    sd["new_emb"] = sd["new_emb"].to(torch.bfloat16)
torch.save(sd, sys.argv[2])
