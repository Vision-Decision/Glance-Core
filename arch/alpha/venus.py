"""VenusBench-GD under the GUI-path alpha (v1.1): positive click accuracy, refusal N-acc at P(present)=0.5, and N-acc at
the threshold that keeps alpha=1's positive acceptance rate.  python arch/alpha/venus.py DIR(a1/, a0.7/, a0.5/) OUT.json
(own scorer; the official UI-Venus evaluator was not re-run, so these are not official Overall numbers)."""
import gzip, json, os, sys
import numpy as np
out = {}
base_rec = None
for a in sorted(os.listdir(sys.argv[1]), key=lambda s: -float(s[1:])):
    R = [json.loads(l) for l in gzip.open(f"{sys.argv[1]}/{a}/venus.rows.jsonl.gz", "rt")]
    P = [r for r in R if r["present"]]; A = [r for r in R if not r["present"]]
    hit = np.array([r["gt"][0] <= (r["box"][0] + r["box"][2]) / 2 <= r["gt"][2] and r["gt"][1] <= (r["box"][1] + r["box"][3]) / 2 <= r["gt"][3] for r in P], float)
    pp = np.array([r["p_logit"] for r in P]); pa = np.array([r["p_logit"] for r in A])
    rec = float((pp > 0).mean()); base_rec = base_rec if base_rec is not None else rec
    t = np.quantile(pp, 1 - base_rec)
    out[a] = dict(n_pos=len(P), n_abs=len(A), pos_click=float(hit.mean()), refusal_nacc_0=float((pa < 0).mean()),
                  pos_accept_0=rec, refusal_nacc_at_a1_accept=float((pa < t).mean()))
    print(a, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[a].items()})
json.dump(out, open(sys.argv[2], "w"), indent=1)
