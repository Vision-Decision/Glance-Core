"""glance-core results. Sources (all in glance-core):
 RefCOCO val Acc@0.5 / gRefCOCO no-target accuracy:
   Glance v1  outputs/v1/eval_results.json sets.refcoco_val.acc50 = 0.9317, sets.grefcoco_val_merged.N_acc = 0.8740
   Gen-v1     outputs/gen_v1/summary.md (RefCOCO val 93.59, N-acc 88.37)
   base Qwen3.5-4B generative, 500-item samples: outputs/v0/baseline_results.json (acc50 0.914, N_acc 0.65)
 ScreenSpot-Pro (1,581 items, native resolution), Glance v1.1 checkpoint:
   outputs/arch/zoom/results.json: one pass acc1 0.5009; alpha=0.5 acc1 0.5781; alpha=0.5 + in-block zoom
   (sspro_p16384_c4096_pre_alliso_k1.0_a0.5) acc2 0.6875;  base 59.9: outputs/sspro_base/summary.md
 VenusBench-GD official UI-Venus scorer overall: Glance v1 40.06 (outputs/sprint_abs/v1/tables.md raw_gated),
   Gen-v1 28.18 (outputs/gen_v1/summary.md), Qwen3-VL-4B 56.15 reported (outputs/sprint_abs/literature.md)."""
import matplotlib.pyplot as plt
from figstyle import render

PANELS = [
    ("Rejects absent objects: far above the base, level with text", "gRefCOCO val, no-target accuracy (%)", [
        ("Qwen3.5-4B base, text (500-item sample)", 65.0, "base"),
        ("Gen-v1: same data, text answers", 88.37, "gen"),
        ("Glance v1 (released)", 87.40, "glance")], (0, 100)),
    ("Box accuracy: on par with text answers", "RefCOCO val, Acc@0.5 (%)", [
        ("Qwen3.5-4B base, text (500-item sample)", 91.4, "base"),
        ("Gen-v1: same data, text answers", 93.59, "gen"),
        ("Glance v1 (released)", 93.17, "glance")], (0, 100)),
    ("GUI clicks: training-free steps on one checkpoint", "ScreenSpot-Pro, native res.; zoom track, settings chosen on test (%)", [
        ("Qwen3.5-4B base, text answers", 59.9, "base"),
        ("Glance v1.1, one pass", 50.09, "glance_dim"),
        ("+ GUI LoRA scale α = 0.5", 57.81, "glance_dim"),
        ("+ in-block zoom (test-tuned)", 68.75, "glance")], (0, 100)),
    ("GUI grounding: a general 4B VLM still leads", "VenusBench-GD, official scorer, overall", [
        ("Gen-v1: same data, text answers", 28.18, "gen"),
        ("Glance v1 (released)", 40.06, "glance"),
        ("Qwen3-VL-4B (reported)", 56.15, "lit")], (0, 100)),
]

LABEL = {68.75: "68.75", 56.15: "56.15", 40.06: "40.06", 28.18: "28.18"}

def draw(t, mode):
    col = {"base": t["orange"], "gen": t["aqua"], "glance": t["blue"], "glance_dim": t["blue_t"], "lit": t["gray"]}
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=1.05, left=0.215, right=0.965, top=0.86, bottom=0.07,
                          height_ratios=[3, 4])
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (title, sub, rows, lim), (r, c) in zip(PANELS, order):
        ax = fig.add_subplot(gs[r, c])
        n = len(rows)
        for i, (lab, v, k) in enumerate(rows):
            y = n - 1 - i
            ec = t["blue"] if k == "glance_dim" else "none"
            ax.barh(y, v, height=0.62, color=col[k], edgecolor=ec, lw=1.2)
            ax.text(v + 1.2, y, LABEL.get(v, f"{v:.1f}"), va="center", fontsize=10,
                    color=t["ink"], weight="bold" if k == "glance" else "normal")
        lits = [n - 1 - i for i, r in enumerate(rows) if r[2] == "lit"]
        if lits and len(lits) < n:
            ax.axhline(max(lits) + 0.5, color=t["axis"], lw=1)
        ax.set_yticks(range(n)); ax.set_yticklabels([x[0] for x in rows][::-1], fontsize=9.5)
        for tl, (_, _, k) in zip(ax.get_yticklabels(), rows[::-1]):
            tl.set_color(t["blue"] if k.startswith("glance") else t["ink"])
            if k == "glance": tl.set_weight("bold")
        ax.set_xlim(*lim); ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color=t["grid"], lw=1); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="x", labelsize=9)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color=t["ink"], pad=20, x=-0.0)
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9.5, color=t["ink2"], va="bottom")
    from matplotlib.patches import Patch
    hs = [Patch(color=col["glance"], label="Glance"), Patch(color=col["gen"], label="Gen-v1: same backbone and data, answers in text"),
          Patch(color=col["base"], label="Qwen3.5-4B base, answers in text"), Patch(color=col["lit"], label="other models, reported numbers")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=4, frameon=False, fontsize=10, labelcolor=t["ink"])
    return fig

if __name__ == "__main__":
    render("core-results", draw)
