"""Launch-style benchmark table (rows grouped by category, our model first and outlined in blue).
Cell colour: blue = our model is best in the row; grey = a baseline is best. A row is coloured only when at least
two models report it; ties at the displayed precision colour every tied cell. '—' = not reported/measured.
Usage: python bench_table.py SPEC.json OUTNAME   (writes OUTNAME-{light,dark}.{svg,png})."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

T = {
    "light": dict(bg="#ffffff", ink="#1f1f1f", ink2="#5f6368", rule="#dadce0", strong="#3c4043",
                  ours_fill="#a8d1ff", ours_edge="#4c8df6", other_fill="#e3e5e8", head_ours="#ffffff"),
    "dark": dict(bg="#0d1117", ink="#e8eaed", ink2="#9aa0a6", rule="#30363d", strong="#9aa0a6",
                 ours_fill="#1c4a80", ours_edge="#5a9cf8", other_fill="#2d333b", head_ours="#0d1117"),
}

def fmt(v, f):
    return "—" if v is None else f.format(v)

def winners(vals, f, better):
    shown = [(i, fmt(v, f)) for i, v in enumerate(vals) if v is not None]
    if len(shown) < 2:
        return set()
    key = lambda s: float(s.replace(",", "").rstrip("%"))
    best = (max if better == "high" else min)(key(s) for _, s in shown)
    return {i for i, s in shown if key(s) == best}

def draw(spec, mode):
    t = T[mode]
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "svg.fonttype": "path"})
    cols = spec["columns"]; rows = spec["rows"]
    W_cat, W_bench, W_note, W_col = 1.55, 2.75, 1.55, 1.75
    W = W_cat + W_bench + W_note + W_col * len(cols)
    rh, hh, foot = 0.46, 0.62, 0.35 + 0.22 * len(spec["footer"])
    H = hh + rh * len(rows) + foot + 0.15
    fig = plt.figure(figsize=(W + 0.4, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(-0.2, W + 0.2); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor(t["bg"] if spec.get("opaque") else "none")
    x0 = W_cat + W_bench + W_note
    cx = [x0 + W_col * (j + 0.5) for j in range(len(cols))]
    # header
    ax.text(W_cat + 0.05, hh / 2, "Benchmark", va="center", fontsize=12, color=t["ink"])
    for j, c in enumerate(cols):
        ax.text(cx[j], hh / 2, c, va="center", ha="center", fontsize=11.5 if j else 12,
                weight="bold" if j == 0 else "normal", color=t["ink"])
    ax.plot([0, W], [hh, hh], color=t["strong"], lw=1.0)
    y = hh
    prev_cat = None
    for r, row in enumerate(rows):
        cat = row.get("cat")
        if cat and cat != prev_cat and r > 0:
            ax.plot([0, W], [y, y], color=t["strong"], lw=0.9)
        elif r > 0:
            ax.plot([W_cat, W], [y, y], color=t["rule"], lw=0.8)
        if cat and cat != prev_cat:
            ax.text(0.05, y + 0.12, cat, va="top", fontsize=10.5, color=t["ink2"], linespacing=1.2)
        prev_cat = cat or prev_cat
        f = row.get("fmt", "{:.1f}")
        win = winners(row["vals"], f, row.get("better", "high"))
        for j, v in enumerate(row["vals"]):
            if j in win:
                ax.add_patch(Rectangle((x0 + W_col * j, y), W_col, rh, fc=t["ours_fill"] if j == 0 else t["other_fill"],
                                       ec="none", zorder=0))
            mark = (row.get("marks") or [""] * len(cols))[j] or ""
            s = fmt(v, f)
            ax.text(cx[j], y + rh / 2, s, va="center", ha="center", fontsize=11.5,
                    weight="bold" if j in win else "normal", color=t["ink"])
            if mark and v is not None:
                ax.text(cx[j] + 0.045 * len(s) + 0.07, y + rh / 2 - 0.09, mark, va="center", ha="left", fontsize=8,
                        color=t["ink2"])
        ax.text(W_cat + 0.05, y + rh / 2, row["bench"], va="center", fontsize=11, color=t["ink"])
        if row.get("note"):
            ax.text(x0 - 0.08, y + rh / 2, row["note"], va="center", ha="right", fontsize=8.8, color=t["ink2"],
                    linespacing=1.1)
        y += rh
    ax.plot([0, W], [y, y], color=t["strong"], lw=1.0)
    ax.add_patch(Rectangle((x0, 0.04), W_col, y - 0.04, fc="none", ec=t["ours_edge"], lw=2.0, zorder=5))
    for k, line in enumerate(spec["footer"]):
        ax.text(0.0, y + 0.3 + 0.22 * k, line, va="center", fontsize=8.3, color=t["ink2"])
    return fig

if __name__ == "__main__":
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    for mode in ("light", "dark"):
        fig = draw(spec, mode)
        fig.savefig(f"{out}-{mode}.svg", transparent=not spec.get("opaque"), metadata={"Date": None})
        fig.savefig(f"{out}-{mode}.png", dpi=150, facecolor=T[mode]["bg"])
        plt.close(fig)
