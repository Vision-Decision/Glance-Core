"""Markdown table (kind x lang) from eval_diag.py outputs.  python3 scripts/ground_v2/diag_table.py [dir/]"""
import json, sys
D = sys.argv[1] if len(sys.argv) > 1 else "outputs/ground_v2/diag/"
r = {m: json.load(open(D + m + ".json"))["per_kind"] for m in ["v1", "v1.1"]}
order = ["short", "medium", "long", "casual", "cat_only", "pos_pos_sole", "function@1", "group", "cat_all", "function@multi",
         "gref_multi", "hneg_attribute", "hneg_relation", "neg_count", "neg_position", "neg_near"]
def fmt(v):
    if "n_acc" in v: return f"{v['n_acc']}"
    if "acc50" in v: return f"{v['acc50']} (rec {v['recall']})"
    return f"rec {v['recall']} / hit {v['hit_any']} / ∪ {v['union50']}"
print("| kind | metric | n(en/zh) | v1 en | v1 zh | v1.1 en | v1.1 zh |")
print("|---|---|---|---|---|---|---|")
for k in order:
    rows = [r[m].get(f"{k}/{l}") for m in ["v1", "v1.1"] for l in ["en", "zh"]]
    v = next(x for x in rows if x)
    met = "N-acc" if "n_acc" in v else ("Acc@0.5 gated (recall)" if "acc50" in v else "recall / hit-any / union")
    n = "/".join(str(r["v1"].get(f"{k}/{l}", {}).get("n", 0)) for l in ["en", "zh"])
    print(f"| {k} | {met} | {n} | " + " | ".join(fmt(x) if x else "–" for x in rows) + " |")
