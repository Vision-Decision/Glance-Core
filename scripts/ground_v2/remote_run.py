"""Run a command on gen-host with secrets in its environment only.

Secrets are read locally (OLLAMA_API_KEY from the repo .env, HF_TOKEN from ~/.cache/huggingface/token) and sent over
ssh stdin; they are never echoed, put on a command line or written to disk on either side.
  python3 scripts/ground_v2/remote_run.py [--fg] [--env OLLAMA_API_KEY,HF_TOKEN] LOGFILE -- python scripts/ground_v2/x.py ...
Relative paths are under ~/gv2/code on the host; `python` = ~/gv2/.venv/bin/python. Default: background (nohup).
"""
import os, shlex, subprocess, sys

args = sys.argv[1:]
fg = "--fg" in args
if fg:
    args.remove("--fg")
names = ["OLLAMA_API_KEY"]
if "--env" in args:
    i = args.index("--env"); names = args[i + 1].split(","); del args[i:i + 2]
log, cmd = args[0], args[args.index("--") + 1:]


def secret(name):
    if name == "HF_TOKEN":
        return open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    if not os.path.exists(env):  # worktree: fall back to the main checkout
        env = os.path.expanduser("~/Developer/decision-vison/.env")
    return next(l.split("=", 1)[1].strip().strip('"').strip("'") for l in open(env) if l.startswith(name + "="))


vals = [secret(n) for n in names]
if cmd[0] == "python":
    cmd[0] = "../.venv/bin/python"
reads = "; ".join(f"read -r {n}; export {n}" for n in names)
run = " ".join(shlex.quote(c) for c in cmd)
remote = f"{reads}; cd ~/gv2/code; " + (f"{run} > {shlex.quote(log)} 2>&1 < /dev/null; echo exit $?" if fg else
                                         f"nohup {run} > {shlex.quote(log)} 2>&1 < /dev/null & echo started $!")
p = subprocess.run(["ssh", "gen-host", remote], input="".join(v + "\n" for v in vals), text=True, capture_output=True)
print(p.stdout.strip(), p.stderr.strip()[-300:])
