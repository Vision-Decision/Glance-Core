#!/usr/bin/env bash
# Run a shell command on a pooled Colab VM and print its output:  scripts/rsh.sh SESSION "cmd"
S=${1:?session}; shift
python3 - "$*" <<'PY' | timeout ${RSH_WALL:-900} colab-pool exec "$S" --timeout ${RSH_TIMEOUT:-1800} 2>&1 | grep -av '^\[pool\]'
import sys
print("import subprocess\nr=subprocess.run(%r,shell=True,capture_output=True,text=True)\nprint(r.stdout[-20000:]);print(r.stderr[-5000:])" % sys.argv[1])
PY
