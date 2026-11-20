"""Fair-latency report: table + plot from outputs/ablation_v2/latency/{ours_ar,vllm_pc,vllm_nopc}.json.

python ablation/latency_v2_report.py [--dir outputs/ablation_v2/latency]
Writes <dir>/latency_table.md, <dir>/latency_vs_N.png, <dir>/latency_summary.json.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NS = [1, 2, 4, 8, 16, 32]


def load(d):
    sys = {}
    p = f"{d}/ours_ar.json"
    if os.path.exists(p):
        r = json.load(open(p))
        sys["Glance (ours)"] = r.get("ours", {})
        sys["AR-digits, incremental KV (our engine)"] = r.get("ar_digits_inc", {})
        sys["AR presence-only q(t1=none), grouped blocks"] = r.get("ar_digits_presence", {})
    p = f"{d}/presence_1pass.json"
    if os.path.exists(p):
        sys["AR presence-only q(t1=none), one padded pass"] = json.load(open(p))["ar_digits_presence_1pass"]
    for f, lab in (("vllm_pc", "Qwen3.5-4B generative, vLLM + prefix cache"),
                   ("vllm_nopc", "Qwen3.5-4B generative, vLLM, no prefix cache")):
        p = f"{d}/{f}.json"
        if os.path.exists(p):
            m = json.load(open(p))["modes"]
            sys[lab + " (all requests at once)"] = {k: v for k, v in m.items() if not k.endswith("_staged")}
            st = {k[:-7]: v for k, v in m.items() if k.endswith("_staged")}
            if st:
                st.setdefault("bs1_N1", m.get("bs1_N1")); st.setdefault("bs8_N1", m.get("bs8_N1"))
                sys[lab + " (staged: prefix first)"] = {k: v for k, v in st.items() if v}
    return sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/ablation_v2/latency")
    a = ap.parse_args()
    S = load(a.dir)
    L = ["| system | " + " | ".join(f"N={n}" for n in NS) + " | bs8 N=1 | bs8 N=8 |", "|---|" + "---|" * (len(NS) + 2)]
    for name, m in S.items():
        cells = []
        for n in NS:
            v = m.get(f"bs1_N{n}")
            cells.append(f"{v['ms_median']:.0f} / {v['ms_p90']:.0f}" if v else "–")
        for k in ("bs8_N1", "bs8_N8"):
            v = m.get(k)
            cells.append(f"{v['ms_median']:.0f} ({v['q_per_s']:.0f} q/s)" if v else "–")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "Cells: per-call latency median / p90 in ms (one image, N questions); bs8 columns: 8 images per call, "
          "median ms and throughput in questions/s.", "",
          "| system | " + " | ".join(f"TTFA N={n}" for n in NS) + " | " + " | ".join(f"q/s N={n}" for n in NS) + " |",
          "|---|" + "---|" * (2 * len(NS))]
    for name, m in S.items():
        t = [f"{m[f'bs1_N{n}']['ttfa_ms_median']:.0f}" if m.get(f"bs1_N{n}") else "–" for n in NS]
        q = [f"{m[f'bs1_N{n}']['q_per_s']:.1f}" if m.get(f"bs1_N{n}") else "–" for n in NS]
        L.append(f"| {name} | " + " | ".join(t) + " | " + " | ".join(q) + " |")
    open(f"{a.dir}/latency_table.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    summ = {name: {k: {kk: v[kk] for kk in ("ms_median", "ms_p90", "q_per_s", "ttfa_ms_median") if kk in v}
                   for k, v in m.items()} for name, m in S.items()}
    json.dump(summ, open(f"{a.dir}/latency_summary.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cols = ["#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#7b1fa2", "#9aa0a6", "#5f6368"]
    for (name, m), c in zip(S.items(), cols):
        xs = [n for n in NS if m.get(f"bs1_N{n}")]
        axes[0].plot(xs, [m[f"bs1_N{n}"]["ms_median"] for n in xs], "o-", color=c, label=name, lw=1.6, ms=4)
        axes[1].plot(xs, [m[f"bs1_N{n}"]["q_per_s"] for n in xs], "o-", color=c, label=name, lw=1.6, ms=4)
    for ax, t in ((axes[0], "per-call latency, median (ms)"), (axes[1], "throughput (questions/s)")):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
        ax.set_xlabel("questions per image (N), one image per call"); ax.set_title(t, loc="left"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{a.dir}/latency_vs_N.png", dpi=150)


if __name__ == "__main__":
    main()
