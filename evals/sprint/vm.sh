#!/usr/bin/env bash
# Laptop-side helpers for the GUI sprint VMs (all remote calls have hard timeouts).
#   evals/sprint/vm.sh push NAME          pack dv/ evals/ scripts/ configs/ + entry points, unpack at /content/dv
#   evals/sprint/vm.sh bg NAME LOG "cmd"  launch a detached command on the VM (stdout+stderr -> LOG)
#   evals/sprint/vm.sh sh NAME "cmd"      run a short shell command and print its output
cd "$(dirname "$0")/../.."
TMP=${TMPDIR:-/tmp}
case $1 in
push) COPYFILE_DISABLE=1 tar czf $TMP/sg_code.tgz --exclude=__pycache__ dv train.py eval.py evals scripts configs 2>/dev/null
      timeout 900 colab-pool upload $2 $TMP/sg_code.tgz /content/code.tgz 2>&1 | grep -v '^\[pool\]' | cut -c1-60
      RSH_WALL=900 RSH_TIMEOUT=120 scripts/rsh.sh $2 "mkdir -p /content/dv && tar xzf /content/code.tgz -C /content/dv 2>/dev/null; echo pushed" ;;
bg)   CMD="$4" LOG="$3" python3 scripts/remote_bg.py | timeout 900 colab-pool exec $2 --timeout 120 2>&1 | grep -v '^\[pool\]' ;;
run)  S=$2; LOG=$3; SCRIPT=$4; shift 4   # vm.sh run NAME LOG evals/sprint/x.sh args...  (detached)
      CMD="cd /content/dv && bash $SCRIPT $*" LOG="$LOG" python3 scripts/remote_bg.py | timeout 900 colab-pool exec $S --timeout 120 2>&1 | grep -v '^\[pool\]' ;;
sh)   RSH_WALL=${WALL:-900} RSH_TIMEOUT=${TO:-300} scripts/rsh.sh $2 "$3" ;;
esac
