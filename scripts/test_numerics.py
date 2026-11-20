"""bf16 noise floor vs our shared-prefix forward, real weights, one image + 3 blocks (cosine sim per position)."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from dv.config import Config
from dv.data.dataset import RecordDataset, collate
from dv.model.backbone import SharedPrefixLM
mid = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
proc = AutoProcessor.from_pretrained(mid)
hf = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
tmp = tempfile.mkdtemp(); recs = []
for i, (W, H) in enumerate([(320, 240), (200, 300)]):
    Image.fromarray((np.random.RandomState(i).rand(H, W, 3) * 255).astype("uint8")).save(f"{tmp}/im{i}.jpg")
    recs.append(dict(file=f"im{i}.jpg", W=W, H=H, questions=[dict(text="the red car on the left", present=True, box=[1, 2, 50, 60]),
                                                           dict(text="a dog", present=False, box=None)]))
open(f"{tmp}/r.jsonl", "w").write("\n".join(json.dumps(r) for r in recs))
cfg = Config().update(dict(attn_mask="causal", max_vis_tokens=64, grad_ckpt=False))
ds = RecordDataset(f"{tmp}/r.jsonl", tmp, proc, cfg, train=False)
batch = collate([ds[0], ds[1]], ds.tok); batch.pop("meta")
batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
def ref(m_idx):
    b = int(batch["q_img"][m_idx]); pv = batch["p_valid"][b]
    ids = torch.cat([batch["p_ids"][b][pv], batch["q_ids"][m_idx][batch["q_valid"][m_idx]]])[None]
    n0 = int(batch["grid_thw"][:b].prod(-1).sum()); n1 = int(batch["grid_thw"][b].prod())
    return hf.model(input_ids=ids, pixel_values=batch["pixel_values"][n0:n0 + n1], image_grid_thw=batch["grid_thw"][b:b+1],
                    mm_token_type_ids=(ids == hf.config.image_token_id).int()).last_hidden_state[0].float(), b, pv
def mine():
    net = SharedPrefixLM(hf, cfg).eval()
    feats = net.image_features(batch["pixel_values"], batch["grid_thw"])
    hp, hq = net(batch, feats); return hp.float(), hq.float()
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
with torch.no_grad():
    R16 = [ref(m) for m in range(4)]; P16, Q16 = mine()
    hf.float(); R32 = [ref(m) for m in range(4)]; P32, Q32 = mine()
for m in range(4):
    r16, b, pv = R16[m]; r32 = R32[m][0]; Lq = int(batch["q_valid"][m].sum())
    print(f"q{m}: ref16~ref32 prefix cos min {cos(r16[:-Lq], r32[:-Lq]).min():.4f} q {cos(r16[-Lq:], r32[-Lq:]).min():.4f} | "
          f"mine32~ref32 prefix {cos(P32[b][pv], r32[:-Lq]).min():.5f} q {cos(Q32[m,:Lq], r32[-Lq:]).min():.5f} | "
          f"mine16~ref32 prefix {cos(P16[b][pv], r32[:-Lq]).min():.4f} q {cos(Q16[m,:Lq], r32[-Lq:]).min():.4f}")
