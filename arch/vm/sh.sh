#!/usr/bin/env bash
# Run a shell command on a VM, print output:  arch/vm/sh.sh SESSION "cmd"
S=${1:?session}; shift
python3 - "$*" <<'PY' | timeout ${RSH_WALL:-900} colab-pool exec "$S" 2>&1 | grep -av '^\[pool\]'
import sys
print("import subprocess\nr=subprocess.run(%r,shell=True,capture_output=True,text=True)\nprint(r.stdout[-15000:]);print(r.stderr[-4000:])" % sys.argv[1])
PY
