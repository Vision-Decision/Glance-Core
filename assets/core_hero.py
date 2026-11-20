"""glance-core hero: architecture block diagram.
Backbone facts: Qwen/Qwen3.5-4B-Base config (32 layers: 24 linear_attention (Gated DeltaNet), 8 full_attention);
LoRA r=64 and heads per docs/DESIGN.md and dv/config.py (multibox: heat / match / count heads)."""
from matplotlib.patches import FancyBboxPatch, Polygon
import matplotlib.pyplot as plt
from figstyle import render

def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.8, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

def arrow(ax, p0, p1, t, col=None, lw=1.3):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=col or t["muted"], lw=lw,
                                                      mutation_scale=11, shrinkA=0, shrinkB=0))

def draw(t, mode):
    fig = plt.figure(figsize=(12.8, 5.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 128); ax.set_ylim(0, 56); ax.axis("off")

    # inputs
    ax.text(2, 53.5, "Input", fontsize=11, weight="bold", color=t["ink2"], va="top")
    box(ax, 2, 36, 14, 11, t["card"], t["cardline"], r=0.7)
    ax.add_patch(Polygon([(3.5, 37.5), (7.5, 43.5), (10, 40.5), (11.8, 42.3), (14.5, 37.5)], fc=t["gray"], alpha=.8, zorder=3))
    ax.add_patch(plt.Circle((12, 44.4), 1.0, fc=t["gray"], alpha=.8, zorder=3))
    ax.text(9, 34.6, "one image", ha="center", va="top", fontsize=9, color=t["ink2"])
    ax.text(9, 31.5, "+ N typed questions\nabout it", ha="center", va="top", fontsize=9.5, color=t["blue"], weight="bold")
    # backbone
    ax.text(25, 53.5, "Qwen3.5-4B hybrid backbone + LoRA (r = 64)", fontsize=11, weight="bold", color=t["ink2"], va="top")
    box(ax, 25, 6, 46, 44, t["card"], t["cardline"], r=1.2, z=1)
    # layer pattern strip: 8 x (3 GDN + 1 attention)
    x0 = 27.5
    for i in range(32):
        is_att = (i % 4 == 3)
        ax.add_patch(FancyBboxPatch((x0 + i * 1.28, 41.5), 0.95, 5.5, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc=t["violet"] if is_att else t["aqua"], ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((27.5, 38.2), 1.6, 1.6, boxstyle="round,pad=0,rounding_size=0.3", fc=t["aqua"], ec="none", zorder=3))
    ax.text(29.7, 39.0, "24 Gated DeltaNet layers", fontsize=8.8, color=t["ink"], va="center")
    ax.add_patch(FancyBboxPatch((48.5, 38.2), 1.6, 1.6, boxstyle="round,pad=0,rounding_size=0.3", fc=t["violet"], ec="none", zorder=3))
    ax.text(50.7, 39.0, "8 full-attention layers", fontsize=8.8, color=t["ink"], va="center")
    ax.plot([26.5, 69.5], [35.6, 35.6], color=t["cardline"], lw=1)
    ax.text(27.5, 33.6, "One forward pass over the token layout:", fontsize=9.2, weight="bold", color=t["ink"], va="center")
    # prefix bar
    box(ax, 27.5, 26.5, 18, 4.2, t["gray_t"], t["cardline"], lw=1.0, r=0.5)
    ax.text(36.5, 28.6, "image prefix,\nrun once", ha="center", va="center", fontsize=8.8, color=t["ink"])
    arrow(ax, (16.3, 39.5), (27.3, 29.5), t)
    ys = [30.2, 25.6, 21.0, 14.0]
    for i, y in enumerate(ys):
        ax.plot([45.6, 49.5], [28.6, y + 1.1], color=t["blue"], lw=1.0, zorder=2)
        box(ax, 49.5, y, 19.5, 2.4, t["blue"], t["blue"], lw=0, r=0.4)
        ax.text(50.4, y + 1.2, ["Is there a dog?", "Box: the red mug", "Count the chairs", "Indoor or outdoor?"][i],
                ha="left", va="center", fontsize=8.4, color=t["onfill"], weight="bold")
    ax.text(59.25, 18.4, "⋮", ha="center", va="center", fontsize=10, color=t["muted"])
    ax.text(27.5, 23.4, "GDN layers: every block starts\nfrom the prefix's recurrent state.\n"
            "Attention layers: every block sees\nthe prefix and itself, never\nanother block.",
            fontsize=8.4, color=t["ink2"], va="top", linespacing=1.3)
    ax.text(27.5, 8.4, "question blocks: one per question, all in the same pass", fontsize=9, color=t["blue"], weight="bold", va="center")
    # heads
    ax.text(76, 53.5, "Readout heads (no decoding)", fontsize=11, weight="bold", color=t["ink2"], va="top")
    heads = [("Presence", "P(present): is it there at all?", t["blue"]),
             ("Box", "patch choice + 4 edge distributions → box,\nIoU head → P(IoU ≥ 0.5)", t["blue"]),
             ("Multi-box / count  (variant, in evaluation)", "heat-map peaks + P(n = 1, 2, 3, ≥ 4)", t["blue"]),
             ("Typed", "yes / no, choice, score → a probability\nfor every allowed answer", t["blue"])]
    for i, (k, v, c) in enumerate(heads):
        y = 41 - i * 9.6
        variant = k.startswith("Multi")
        ax.add_patch(FancyBboxPatch((76, y), 50, 8, boxstyle="round,pad=0,rounding_size=0.8", fc=t["card"] if variant else t["blue_t"],
                                    ec=t["blue"], lw=1.1, ls="--" if variant else "-", zorder=2))
        ax.text(78, y + 5.6, k, fontsize=10.5, weight="bold", color=t["blue"], va="center")
        ax.text(78, y + 2.6, v, fontsize=8.8, color=t["ink"], va="center", linespacing=1.25)
        arrow(ax, (69.4, 22), (75.8, y + 4), t, col=t["blue"], lw=1.0)
    ax.text(126, 3.6, "released Glance-4B-preview: presence, box and typed heads",
            ha="right", fontsize=8.5, color=t["ink2"])
    return fig

if __name__ == "__main__":
    render("core-architecture", draw)
