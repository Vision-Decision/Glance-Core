#!/usr/bin/env bash
# queue q_b2 (replaces q_b): waits for r1 to finish, then serial jobs
while pgrep -f "[r]un_recipe.sh" > /dev/null; do sleep 30; done
for j in e_rob_r1 a_nofunc a_nozh d_heads r5_teacher; do bash /content/dv/scripts/sprint_robust/jobs/$j.sh > /content/sr/$j.log 2>&1; done
echo "[q_b2] ALL DONE $(date -u)"
