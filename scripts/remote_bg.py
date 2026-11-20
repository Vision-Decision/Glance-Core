# Launch a shell command detached on the VM (survives exec disconnects). Edit CMD/LOG via env before piping:
#   CMD="cd /content/dv && python train.py ..." LOG=/content/out/x.log python scripts/remote_bg.py | colab-pool exec S
import os, sys
cmd, log = os.environ["CMD"], os.environ["LOG"]
print(f"""import subprocess, os
os.makedirs(os.path.dirname({log!r}), exist_ok=True)
p = subprocess.Popen(["bash", "-lc", {cmd!r} + " > {log} 2>&1"], start_new_session=True)
print("launched pid", p.pid)""")
