#!/usr/bin/env bash
# Launch a detached shell command on a VM:  arch/vm/bg.sh SESSION LOG "cmd"
S=${1:?session}; LOG=${2:?log}; shift 2
python3 - "$*" "$LOG" <<'PY' | colab-pool exec "$S" 2>&1 | grep -av '^\[pool\]'
import sys
cmd, log = sys.argv[1], sys.argv[2]
print(f"""import subprocess, os
os.makedirs(os.path.dirname({log!r}), exist_ok=True)
p = subprocess.Popen(["bash", "-lc", "( " + {cmd!r} + " ) > {log} 2>&1 < /dev/null"], start_new_session=True)
print("launched pid", p.pid)""")
PY
