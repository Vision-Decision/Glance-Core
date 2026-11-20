#!/usr/bin/env bash
# sg-train after ft2: push ft2 final, evaluate it locally, push the eval rows.
cd /content/dv
bash evals/sprint/push_final.sh /content/out/sprint_ft2/ckpt_final.pt sprint_gui/ft2/ckpt_final.pt
bash evals/sprint/eval_local.sh /content/out/sprint_ft2/ckpt_final.pt ft2f "sspro:16384:1 ss2:16384:1"
cd /content/sprint && tar czf /content/sprint_rows_ft2.tgz ft2f_*.rows.jsonl.gz an_ft2f.json sprint_gui_results_ft2f.json *.info.json
cd /content/dv && bash evals/sprint/push_final.sh /content/sprint_rows_ft2.tgz sprint_gui/eval_rows_ft2.tgz
echo "[q] ALL DONE ft2_final $(date -u +%T)"
