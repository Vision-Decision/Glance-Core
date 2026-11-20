"""Debug probe: launch one command on a fresh X session, dump AT-SPI + X stack + screenshot.
   dbus-run-session -- python probe.py --cmd "firefox" --app firefox --out /tmp/probe"""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from session import XSession, kill_tree

ap = argparse.ArgumentParser()
ap.add_argument("--cmd"); ap.add_argument("--app"); ap.add_argument("--out")
ap.add_argument("--display", type=int, default=90); ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080)
ap.add_argument("--scale", type=int, default=1); ap.add_argument("--wait", type=float, default=15)
ap.add_argument("--keys", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
xs = XSession(a.display, a.W, a.H, a.scale); xs.start()
if a.scale == 2:
    os.environ.update(GDK_SCALE="2", QT_SCALE_FACTOR="2")
p = subprocess.Popen(a.cmd, shell=True, start_new_session=True, stdout=open(f"{a.out}/app.log", "w"), stderr=subprocess.STDOUT)
time.sleep(a.wait)
subprocess.run("wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz", shell=True)
for k in [k for k in a.keys.split(",") if k]:
    if k.startswith("click:"):
        x, y = k[6:].split(":"); subprocess.run(["xdotool", "mousemove", x, y, "click", "1"])
    else:
        subprocess.run(["xdotool", "key", k])
    time.sleep(1.5)
time.sleep(2)
import a11y, filt
from Xlib import display as xd
names = [x.get_name() for x in a11y.apps()]
app = a11y.find_app(a.app)
t = time.time()
nodes, info = a11y.dump_app(app) if app else ([], {})
dt = time.time() - t
disp = xd.Display()
st = a11y.x_stack(disp)
img = a11y.grab(disp, a.W, a.H); img.save(f"{a.out}/shot.png")
kept, fs = filt.filter_elements(nodes, st, a.W, a.H, img)
json.dump(dict(apps=names, info=info, dump_s=dt, xstack=st, nodes=nodes, kept=kept, fstats=fs), open(f"{a.out}/dump.json", "w"))
print("apps", names, "nodes", len(nodes), "dump_s", round(dt, 2), "xstack", [(x[1], x[2], x[3], x[4]) for x in st])
print("fstats", fs)
from collections import Counter
print("roles kept", Counter(e["role"] for e in kept).most_common(15))
kill_tree(p); xs.stop()
