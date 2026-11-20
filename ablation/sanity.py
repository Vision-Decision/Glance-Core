"""On-GPU sanity checks for the AR path: training forward == prefill+block, question states independent of answer."""
import torch
from ablation.common import ABConfig, ARDataset, collate_ar, build_ar
from ablation.train_ab import to_dev, ar_loss

cfg = ABConfig().update(dict(variant="ar_digits", grad_ckpt=False))
for variant in ["ar_digits", "ar_atomic"]:
    cfg = cfg.update(dict(variant=variant))
    net, fmt, proc = build_ar(cfg, "cuda")
    for n, p in net.hf.named_parameters():      # make LoRA non-trivial
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=1e-3)
    net.eval()
    ds = ARDataset(cfg.train_file, cfg.image_root, proc, cfg, limit=4, fmt=fmt, train=False)
    items = [ds[0], ds[1]]
    b = to_dev({k: v for k, v in collate_ar(items, ds.tok).items() if k != "meta"}, "cuda")
    with torch.no_grad():
        feats = net.image_features(b["pixel_values"], b["grid_thw"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, hq = net(b, feats)
            pc = net.prefill(b, feats)
            hq2 = net.block(pc, b["q_ids"], b["q_valid"], b["q_pos"], b["q_img"], b["q_last"])
        sel = b["labels"] != -100
        l1, l2 = net.logits(hq[sel]).float(), net.logits(hq2[sel]).float()
        print(variant, "joint vs prefill+block: max|dlogit|", (l1 - l2).abs().max().item(),
              "argmax agree", (l1.argmax(-1) == l2.argmax(-1)).float().mean().item(), "logit scale", l1.abs().max().item())
        b0 = to_dev({k: v for k, v in collate_ar(items, ds.tok, with_answer=False).items() if k != "meta"}, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hq3 = net.block(pc, b0["q_ids"], b0["q_valid"], b0["q_pos"], b0["q_img"], b0["q_last"])
        M = hq.shape[0]
        a = hq2[torch.arange(M), b["q_last"]].float(); c = hq3[torch.arange(M), b0["q_last"]].float()
        print(variant, "readout with vs without answer appended: max|d|", (a - c).abs().max().item(), "scale", a.abs().max().item())
    net.train()
    loss, parts = ar_loss(net, b)
    print(variant, "loss", loss.item(), {k: float(v) for k, v in parts.items()})
    del net; torch.cuda.empty_cache()
