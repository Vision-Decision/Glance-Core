"""One prefill, many questions: median latency for 1 vs 8 box questions on one COCO image, RTX PRO 6000.
Glance v1 (research forward, eval.py --latency): outputs/v1/eval_results.json latency.box_1q / box_8q ms_median.
Qwen3.5-4B generative (HF generate, no serving engine): outputs/v0/baseline_results.json
sets.refcoco_val.latency_ms_median_1q and latency_ms_median_8q_same_image."""
import matplotlib.pyplot as plt
from figstyle import render

def draw(t, mode):
    fig, ax = plt.subplots(figsize=(9.4, 3.0))
    fig.subplots_adjust(left=0.4, right=0.9, top=0.8, bottom=0.2)
    rows = [("Glance v1, 1 question", 85.53, t["blue"]), ("Glance v1, 8 questions", 85.63, t["blue"]),
            ("Qwen3.5-4B generative, 1 question", 399.60, t["orange"]),
            ("Qwen3.5-4B generative, 8 questions (8 decodes)", 3161.25, t["orange"])]
    for i, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.6, color=c)
        ax.text(v + 40, y, f"{v:,.0f} ms", va="center", fontsize=10.5, color=t["ink"],
                weight="bold" if "Glance" in lab else "normal")
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10, color=t["ink"])
    ax.set_xlim(0, 3600); ax.set_xticks([0, 1000, 2000, 3000]); ax.set_xticklabels(["0", "1 s", "2 s", "3 s"])
    ax.grid(axis="x", color=t["grid"], lw=1); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_title("Eight questions cost the same as one", loc="left", fontsize=12, weight="bold", color=t["ink"], pad=8)
    return fig

if __name__ == "__main__":
    render("core-one-prefill", draw)
