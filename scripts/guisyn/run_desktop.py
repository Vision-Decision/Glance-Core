"""Orchestrator: keep N worker slots busy with (app, resolution, scale) jobs until the time budget is used.

  python run_desktop.py --slots 26 --hours 6 --apps all --steps 90
Each job runs as the unprivileged `guisyn` user in its own Xvfb display + D-Bus session (no network listeners).
"""
import argparse, json, os, random, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apps import APPS

RES = [((1920, 1080, 1), 0.22), ((2560, 1440, 1), 0.33), ((3840, 2160, 1), 0.25), ((3840, 2160, 2), 0.12),
       ((2560, 1600, 1), 0.08)]
CODE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=26)
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--apps", default="all")
    ap.add_argument("--weights", default="")  # app=w,...
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--max_s", type=int, default=1100)
    ap.add_argument("--jobs", type=int, default=0, help="stop after this many jobs (0 = time budget only)")
    ap.add_argument("--res", default="", help="force WxHxS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--display_base", type=int, default=100)
    a = ap.parse_args()
    apps = list(APPS) if a.apps == "all" else a.apps.split(",")
    w = {k: 1.0 for k in apps}
    for kv in filter(None, a.weights.split(",")):
        k, v = kv.split("="); w[k] = float(v)
    rng = random.Random(a.seed or int(time.time()))
    t_end = time.time() + a.hours * 3600
    running = {}  # slot -> (Popen, job desc)
    log = open("/data/guisyn/logs/run_desktop.log", "a")
    n_jobs, counts = 0, {k: 0 for k in apps}
    rr = []
    while True:
        for s, (p, d) in list(running.items()):
            if p.poll() is not None:
                del running[s]
                print(time.strftime("%F %T"), "end", s, d, p.returncode, file=log, flush=True)
        if os.path.exists("/data/guisyn/STOP"):
            if not running:
                break
            time.sleep(5); continue
        launching = time.time() < t_end and (not a.jobs or n_jobs < a.jobs)
        if not launching and not running:
            break
        for s in range(a.slots):
            if not launching or s in running or (a.jobs and n_jobs >= a.jobs):
                continue
            if a.jobs and a.jobs <= len(apps) * 2:  # smoke test: one job per app in order
                if not rr:
                    rr = list(apps)
                app = rr.pop(0)
            else:  # least-served app first (weighted), with jitter
                app = min(apps, key=lambda k: counts[k] / w[k] + rng.random() * 0.5)
            counts[app] += 1
            if a.res:
                W, H, S = map(int, a.res.split("x"))
            else:
                (W, H, S), = rng.choices([r for r, _ in RES], weights=[p for _, p in RES])
            disp = a.display_base + s
            seed = rng.randint(1, 10 ** 9)
            cmd = (f"sudo -u guisyn bash -c 'umask 002; cd {CODE} && exec timeout -k 30 {a.max_s + 240} dbus-run-session -- "
                   f"/data/guisyn/venv/bin/python desktop_worker.py --app {app} --display {disp} --W {W} --H {H} "
                   f"--scale {S} --steps {a.steps} --max_s {a.max_s} --seed {seed}'")
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            running[s] = (p, f"{app} {W}x{H}s{S}")
            n_jobs += 1
            print(time.strftime("%F %T"), "start", s, app, W, H, S, file=log, flush=True)
            time.sleep(1.0)
        time.sleep(3)
    print(time.strftime("%F %T"), "orchestrator done", n_jobs, file=log, flush=True)


if __name__ == "__main__":
    main()
