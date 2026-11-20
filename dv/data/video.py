"""Video prefixes for the typed decision model (Glance-Video).

A video record is a v1 record (dv/data/mixed.py) whose state is a short clip instead of an image:
  {"src": "...", "video": "gv/clips/k700val/abc.mp4" | [path, path, ...], "frames": [i, ...] | [[k, i], ...],
   "dt": 0.5, "max_vis": 96, "plen": <est. prefix tokens>, "text": optional state text after the clip,
   "questions": [typed questions only (choice / noul / score)]}
- `video` paths are relative to cfg.data_root; clips are stored at a fixed frame rate (8 fps in our shards).
- `frames` selects decoded frames of the stored clip(s) in presentation order. This is how the self-supervised
  generators work without re-encoding: reversal = reversed indices, speed change = strided indices, order swap =
  permuted halves, synthetic repetition = repeated index runs, stitching = (clip k, frame i) pairs.
- `dt` = seconds between presented frames (drives the "<t seconds>" timestamps, as Qwen3.5's processor would).

Prompt layout (identical to the HF Qwen3.5 processor for a chat message holding one video, then the question):
  <|im_start|>user\n<|vision_start|> {<t0 seconds><|vision_start|><|video_pad|>*n<|vision_end|>} x T/2 <|vision_end|> [text]
Each temporal group (2 frames, temporal_patch_size=2) becomes one block of gh*gw tokens with 3D positions
(t=h=w offset + spatial grid), exactly like an image; the timestamps are ordinary text tokens.
`scripts/video/test_video_prefix.py` checks the token ids against processor(apply_chat_template(...)) and the
positions against model.model.get_rope_index on the VM.
"""
import math
import numpy as np
import torch

from .dataset import resize_dims

_CACHE = {}


def decode_clip(path, max_cache=8):
    """All frames of a (short, low-res) clip as uint8 [N,H,W,3]. Small per-worker LRU cache."""
    if path in _CACHE:
        return _CACHE[path]
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    arr = np.stack(fr) if fr else np.zeros((0, 2, 2, 3), np.uint8)
    if len(_CACHE) >= max_cache:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[path] = arr
    return arr


def gather_frames(r, root):
    vids = r["video"] if isinstance(r["video"], list) else [r["video"]]
    clips = [decode_clip(f"{root}/{v}") for v in vids]
    out = []
    for f in r["frames"]:
        k, i = (f if isinstance(f, (list, tuple)) else (0, f))
        c = clips[k]
        out.append(c[min(max(int(i), 0), len(c) - 1)])
    H = min(x.shape[0] for x in out); W = min(x.shape[1] for x in out)
    out = [x[:H, :W] for x in out]          # stitched clips may differ by a pixel after scaling
    if len(out) % 2:                          # temporal_patch_size = 2: pad by repeating the last frame (as HF does)
        out.append(out[-1])
    return np.stack(out)


def timestamps(n_frames, dt, merge=2):
    t = [i * dt for i in range(n_frames)]
    return [(t[i] + t[i + merge - 1]) / 2 for i in range(0, n_frames, merge)]


def video_inputs(frames, proc, max_vis, max_upscale=2.0):
    """frames uint8 [T,H,W,3] -> (pixel_values_videos, grid_thw [3]); per-group token count <= max_vis."""
    T, H, W, _ = frames.shape
    w, h = resize_dims(W, H, max_vis, max_upscale)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x.float(), size=(h, w), mode="bicubic", align_corners=False)
    x = x.clamp(0, 255).round().to(torch.uint8)
    vp = proc.video_processor
    out = vp(videos=[x], do_resize=False, do_sample_frames=False, return_tensors="pt")
    return out["pixel_values_videos"], out["video_grid_thw"][0]


def video_segs(r, root, proc, cfg, tok):
    """-> (pixel_values_videos, grid_thw, segs) for a video record; segs follow mixed.py's prefix segment format,
    with ("vid", (gh, gw)) for each temporal group."""
    frames = gather_frames(r, root)
    pv, g = video_inputs(frames, proc, r.get("max_vis") or cfg.video_max_vis, r.get("upscale", cfg.max_upscale))
    T, gh, gw = int(g[0]), int(g[1]) // 2, int(g[2]) // 2
    ts = timestamps(len(frames), float(r.get("dt", 0.5)))
    segs = [("ids", tok.user_open + [tok.vs])]
    for k in range(T):
        segs.append(("ids", tok.enc(f"<{ts[k]:.1f} seconds>") + [tok.vs]))
        segs.append(("vid", (gh, gw)))
        segs.append(("ids", [tok.ve]))
    segs.append(("ids", [tok.ve]))
    return pv, g, segs


def est_prefix_tokens(n_frames, W, H, max_vis, max_upscale=2.0):
    w, h = resize_dims(W, H, max_vis, max_upscale)
    groups = math.ceil(n_frames / 2)
    return 4 + groups * ((w // 32) * (h // 32) + 9)
