#!/usr/bin/env bash
# VM-side: push the latest checkpoint of a run to a PRIVATE HF model repo (overwriting), so nothing goes via the laptop.
# Usage (VM): bash push_ckpt_hf.sh /content/out/v1_1 Vision-Decision/glance-train-ckpts-legacy v1.1 [interval_s]
R=$1; REPO=$2; P=$3; INT=${4:-1200}; last=""
while true; do
  for f in ckpt_final.pt ckpt_last.pt; do
    if [ -f $R/$f ]; then
      m=$(stat -c %Y $R/$f)
      if [ "$f:$m" != "$last" ]; then
        sleep 30
        cp $R/$f /content/push_tmp.pt; [ -f $R/opt_last.pt ] && [ $f = ckpt_last.pt ] && cp $R/opt_last.pt /content/push_opt.pt
        if timeout 1800 hf upload $REPO /content/push_tmp.pt $P/$f --repo-type model --commit-message "$P $f $(date -u +%FT%T)" >/dev/null 2>&1; then
          [ -f /content/push_opt.pt ] && timeout 1800 hf upload $REPO /content/push_opt.pt $P/opt_last.pt --repo-type model >/dev/null 2>&1
          last="$f:$m"; echo "$(date -u +%T) pushed $f"
        else echo "$(date -u +%T) push of $f failed"; fi
        rm -f /content/push_tmp.pt /content/push_opt.pt
      fi
      break
    fi
  done
  [ "${last%%:*}" = ckpt_final.pt ] && { echo "final pushed"; exit 0; }
  sleep $INT
done
