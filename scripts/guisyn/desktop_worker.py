"""One desktop job: fresh Xvfb (W x H, scale) + one app, random-but-safe interactions, capture screenshots + AT-SPI.

  dbus-run-session -- python desktop_worker.py --app inkscape --display 101 --W 2560 --H 1440 --scale 1 \
      --out /data/guisyn/raw --steps 80 --seed 1

Writes (append-only, one closed shard per job):
  {out}/images/{app}/{id}.webp      lossless WebP screenshot
  {out}/a11y/{app}/{id}.json.gz     full AT-SPI tree + X stacking + filter stats (audit)
  {out}/jobs/{job}.jsonl            one raw record per kept screenshot (written as .tmp, renamed when the job ends)
"""
import argparse, gzip, hashlib, json, os, random, re, shutil, subprocess, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session import XSession, kill_tree, THEMES_GTK, ICONS, QT_STYLES
from apps import APPS, KEYS, pick_doc

UNSAFE = re.compile(r"\b(quit|exit|close|log ?out|sign ?(in|out|up)|log ?in|shut ?down|restart|reboot|delete|remove|erase|"
                    r"uninstall|install|update|upgrade|download|upload|send|submit|publish|print|purchase|buy|donate|"
                    r"report|crash|terminal|run|debug|execute|format|trash|clear|reset|revert|capture|connect|share|"
                    r"account|online|website|save|export|render|sync|subscribe|get messages|check for|feedback|"
                    r"render|full ?screen|kiosk|minimi[sz]e|maximi[sz]e|restore|new window|new instance|"
                    r"switch user|lock|suspend|record|start|stop|kill|wipe|purge|overwrite|replace all)\b", re.I)
CLICK_W = {"menu": 6, "menu item": 4, "page tab": 4, "push button": 3, "toggle button": 3, "check box": 1.5,
           "radio button": 1.5, "combo box": 2, "list item": 1.5, "tree item": 1.5, "link": 1, "icon": 1,
           "check menu item": 1, "radio menu item": 0.7, "table cell": 0.5, "split button": 1, "button": 2,
           "push button menu": 1, "tool bar item": 1}
SHIM = "/data/guisyn/shim"


def sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, shell=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode()
    except Exception:
        return ""


class Worker:
    def __init__(self, a):
        self.a = a
        self.rng = random.Random(a.seed)
        self.job = f"{a.app}-{a.W}x{a.H}s{a.scale}-{int(time.time())}-{a.seed}"
        self.home = f"{a.homes}/{a.display}"
        for d in ["images", "a11y", "jobs", "logs"]:
            os.makedirs(f"{a.out}/{d}/{a.app}" if d in ("images", "a11y") else f"{a.out}/{d}", exist_ok=True)
        self.log = open(f"{a.out}/logs/{self.job}.log", "w")
        self.recf = open(f"{a.out}/jobs/{self.job}.jsonl.tmp", "w")
        self.sigs = set()
        self.n_saved = 0
        self.hist = []

    def L(self, *x):
        print(time.strftime("%H:%M:%S"), *x, file=self.log, flush=True)

    # ------------------------------------------------------------- setup
    def setup_env(self):
        a = self.a
        shutil.rmtree(self.home, ignore_errors=True)
        tpl = f"{a.root}/home_tpl/{a.app}"
        if os.path.isdir(tpl):
            shutil.copytree(tpl, self.home, symlinks=True)
        os.makedirs(self.home, exist_ok=True)
        for d in [".ffprofile", ".tbprofile", ".config", ".local/share", ".cache"]:
            os.makedirs(f"{self.home}/{d}", exist_ok=True)
        env = os.environ
        env.update(HOME=self.home, XDG_CONFIG_HOME=f"{self.home}/.config", XDG_DATA_HOME=f"{self.home}/.local/share",
                   XDG_CACHE_HOME=f"{self.home}/.cache", XDG_RUNTIME_DIR=f"{self.home}/.run",
                   PATH=f"{SHIM}:{env.get('PATH', '/usr/bin:/bin')}")
        os.makedirs(f"{self.home}/.run", mode=0o700, exist_ok=True)
        self.theme = self.rng.choice(THEMES_GTK)
        self.icons = self.rng.choice(ICONS)
        self.qtstyle = self.rng.choice(QT_STYLES[:3])
        env["GTK_THEME"] = self.theme
        env["QT_STYLE_OVERRIDE"] = self.qtstyle
        if a.scale == 2:
            env.update(GDK_SCALE="2", GDK_DPI_SCALE="1", QT_SCALE_FACTOR="2", QT_AUTO_SCREEN_SCALE_FACTOR="0")
        for k, v in APPS[a.app].get("env", {}).items():
            env[k] = v
        os.makedirs(f"{self.home}/.config/gtk-3.0", exist_ok=True)
        with open(f"{self.home}/.config/gtk-3.0/settings.ini", "w") as f:
            f.write(f"[Settings]\ngtk-cursor-blink=false\ngtk-icon-theme-name={self.icons}\ngtk-enable-animations=false\n")
        with open(f"{self.home}/.gtkrc-2.0", "w") as f:
            f.write(f'gtk-icon-theme-name="{self.icons}"\ngtk-cursor-blink=0\n')

    def launch(self):
        a = self.a
        spec = APPS[a.app]
        self.doc = pick_doc(a.app, self.rng)
        doc = self.doc
        if doc and os.path.isfile(doc) and not doc.endswith(".url"):  # work on a copy (apps may modify it)
            os.makedirs(f"{self.home}/Documents", exist_ok=True)
            dst = f"{self.home}/Documents/{os.path.basename(doc)}"
            shutil.copy(doc, dst); doc = dst
        elif doc and os.path.isdir(doc):
            dst = f"{self.home}/Documents/{os.path.basename(doc)}"
            shutil.copytree(doc, dst); doc = dst
        elif doc.endswith(".url"):
            doc = open(doc).read().strip()
        cmd = spec["cmd"].format(doc=f"'{doc}'" if doc else "", home=self.home)
        if spec.get("env", {}).get("SAL_USE_VCLPLUGIN"):
            cmd = cmd.replace("soffice ", f"soffice -env:UserInstallation=file://{self.home}/.lo ")
        self.L("launch", cmd, "theme", self.theme, self.icons, self.qtstyle)
        self.proc = subprocess.Popen(cmd, shell=True, start_new_session=True, stdout=self.log, stderr=self.log, cwd=self.home)
        import a11y
        t0 = time.time()
        self.app_acc = None
        while time.time() - t0 < spec["wait"] * 3:
            time.sleep(2)
            acc = a11y.find_app(spec["atspi"])
            if acc is not None:
                try:
                    if acc.get_child_count() > 0:
                        self.app_acc = acc
                        break
                except Exception:
                    pass
        if self.app_acc is None:
            self.L("app did not appear in AT-SPI")
            return False
        time.sleep(spec["wait"] * 0.5)
        if self.rng.random() < 0.8:
            sh("wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz")
        else:
            w, h = int(a.W * self.rng.uniform(0.55, 0.9)), int(a.H * self.rng.uniform(0.55, 0.9))
            x, y = self.rng.randint(0, a.W - w), self.rng.randint(0, a.H - h)
            sh(f"wmctrl -r :ACTIVE: -e 0,{x},{y},{w},{h}")
        time.sleep(5)  # toolkits update child extents lazily after a resize (LibreOffice: ~2-3 s)
        return True

    # ------------------------------------------------------------- capture
    def thumb(self):
        import a11y
        im = a11y.grab(self.disp, self.a.W, self.a.H)
        return im, im.resize((self.a.W // 8, self.a.H // 8)).convert("L")

    @staticmethod
    def diff(t1, t2):
        import numpy as np
        d = np.abs(np.asarray(t1, np.int16) - np.asarray(t2, np.int16))
        return float((d > 12).mean())

    def capture(self):
        import a11y, filt
        a = self.a
        _, t_prev = self.thumb()
        for _ in range(6):  # settle: two consecutive thumbs nearly identical
            time.sleep(0.4)
            _, t = self.thumb()
            if self.diff(t, t_prev) < 0.0005:
                break
            t_prev = t
        acc = self.app_acc
        nodes, info = a11y.dump_app(acc)
        if len(nodes) < 3:  # noqa  # stale handle (app re-registered, e.g. after a first-run dialog in another process)
            acc2 = a11y.find_app(APPS[a.app]["atspi"])
            if acc2 is not None:
                self.app_acc = acc = acc2
                nodes, info = a11y.dump_app(acc)
        xst = a11y.x_stack(self.disp)
        img, t_after = self.thumb()
        if self.diff(t_after, t) > 0.002:
            self.L("unstable screen during dump; skip")
            return None
        if info.get("truncated"):
            self.L("tree truncated", info)
        kept, fst = filt.filter_elements(nodes, xst, a.W, a.H, img)
        objs = info.pop("objs", [])
        hk, failed = [], []
        from collections import Counter
        hc = Counter()
        for e in kept:  # live hit test (state must not have changed: checked below via the screen thumb)
            if e.get("popup"):
                hc["popup"] += 1; e["hit"] = "popup"; hk.append(e); continue
            r = a11y.hit_test(objs, nodes, e, fst["scale"])
            hc[str(r)] += 1
            e["hit"] = r
            # keep: hit lands on the element or inside it; or stops at its direct parent (toolkits that do not
            # hit-test leaf children). Everything else (sibling / unrelated widget / far ancestor) is dropped.
            hk.append(e)  # policy (which hit categories are trusted per toolkit) is applied in build_shards.py
            if r not in ("self", "desc", "anc1", None):
                failed.append([e["role"], e["name"][:60], e["box"], r])
        kept = hk
        fst.update(hit=dict(hc), hit_fail=len(failed), kept=len(kept), hit_fail_els=failed[:80])
        _, t_chk = self.thumb()
        if self.diff(t_chk, t) > 0.002:
            self.L("unstable screen during hit test; skip")
            return None
        from build_shards import hit_ok
        named = [e for e in kept if not e.get("uncaptioned") and hit_ok(e, a.app)]
        if len(named) < 5:
            self.L("too few elements", fst)
            return dict(kept=kept, nodes=nodes, xst=xst, saved=False)
        sig = hashlib.md5(json.dumps(sorted((e["role"], e["name"], int(e["box"][0]) // 24, int(e["box"][1]) // 24)
                                            for e in kept)).encode()).hexdigest()
        if sig in self.sigs:
            return dict(kept=kept, nodes=nodes, xst=xst, saved=False)
        self.sigs.add(sig)
        sid = f"{self.job}-{self.step:04d}"
        ip = f"images/{a.app}/{sid}.webp"
        img.save(f"{a.out}/{ip}", "WEBP", lossless=True, quality=50, method=2)
        ap = f"a11y/{a.app}/{sid}.json.gz"
        with gzip.open(f"{a.out}/{ap}", "wt") as f:
            json.dump(dict(nodes=nodes, xstack=xst, info=info, fstats=fst), f)
        rec = dict(id=sid, app=a.app, job=self.job, W=a.W, H=a.H, scale=a.scale, theme=self.theme, icons=self.icons,
                   qtstyle=self.qtstyle, doc=os.path.basename(self.doc) if self.doc else "", step=self.step,
                   actions=self.hist[-4:], image=ip, a11y=ap, elements=kept, fstats=fst, t=time.time(),
                   all_names=sorted({(n["role"], n["name"]) for n in nodes if n["name"]}))
        self.recf.write(json.dumps(rec) + "\n"); self.recf.flush()
        self.n_saved += 1
        return dict(kept=kept, nodes=nodes, xst=xst, saved=True)

    # ------------------------------------------------------------- actions
    def act(self, cap):
        a, rng = self.a, self.rng
        kept = cap["kept"] if cap else []
        xst = cap["xst"] if cap else []
        clients = [x for x in xst if x[2] is not None and x[2][2] > 50]
        popups = [x for x in xst if x[3] and x[1][2] > 30 and x[1][3] > 30]
        safe = [e for e in kept if e["name"] and not UNSAFE.search(e["name"]) and e["role"] in CLICK_W
                and "sensitive" in e["states"]]
        r = rng.random()
        n_dialog = len(clients) - 1
        if (n_dialog >= 1 and self.dialog_steps > rng.randint(2, 6)) or (popups and rng.random() < 0.25) or \
                (r < 0.08 and (popups or n_dialog >= 1)):
            k = rng.choice([1, 1, 2])
            act = ("escape", k)
            for _ in range(k):
                sh("xdotool key Escape"); time.sleep(0.3)
            if n_dialog >= 1 and self.dialog_steps > 8 and len(clients) > 1:
                top = clients[-1]
                sh(f"wmctrl -ic {top[0]}")
                act = ("close_dialog", top[4])
        elif 0.08 <= r < 0.13 and kept:
            e = rng.choice(kept)
            x, y = self._pt(e)
            sh(f"xdotool mousemove {x} {y}"); time.sleep(1.2)
            act = ("hover", e["name"])
        elif r < 0.24 and kept:
            e = rng.choice(kept)
            x, y = self._pt(e)
            sh(f"xdotool mousemove {x} {y} click 3")
            act = ("rightclick", e["name"])
        elif r < 0.30 and KEYS.get(a.app):
            k = rng.choice(KEYS[a.app])
            sh(f"xdotool key {k}")
            act = ("key", k)
        elif r < 0.35 and kept:
            e = rng.choice(kept)
            x, y = self._pt(e)
            sh(f"xdotool mousemove {x} {y} click {rng.choice([4, 5, 5])} click {rng.choice([4, 5])}")
            act = ("scroll", e["name"])
        elif r < 0.38:
            ents = [e for e in safe + kept if e["role"] in ("entry", "text") and re.search(r"search|find|filter", e["name"], re.I)
                    and a.app != "firefox"]
            if ents:
                e = rng.choice(ents)
                x, y = self._pt(e)
                w = rng.choice(["color", "layer", "font", "grid", "view", "size", "text", "line", "zoom", "edit"])
                sh(f"xdotool mousemove {x} {y} click 1"); time.sleep(0.3)
                sh(f"xdotool type --delay 40 {w}")
                act = ("type", w)
            else:
                act = ("noop",)
        elif safe:
            ws = [CLICK_W[e["role"]] * (2.0 if popups and e["role"].endswith("menu item") else 1.0) for e in safe]
            e = rng.choices(safe, weights=ws)[0]
            x, y = self._pt(e)
            sh(f"xdotool mousemove {x} {y} click 1")
            act = ("click", e["role"], e["name"])
        else:
            sh("xdotool key Escape")
            act = ("escape", 1)
        self.dialog_steps = self.dialog_steps + 1 if n_dialog >= 1 else 0
        self.hist.append(act)
        self.L("act", act)
        time.sleep(rng.uniform(0.6, 1.4))

    def _pt(self, e):
        b = e["box"]
        return int((b[0] + b[2]) / 2 + self.rng.uniform(-0.2, 0.2) * (b[2] - b[0])), \
               int((b[1] + b[3]) / 2 + self.rng.uniform(-0.2, 0.2) * (b[3] - b[1]))

    # ------------------------------------------------------------- main
    def run(self):
        a = self.a
        self.setup_env()
        xs = XSession(a.display, a.W, a.H, a.scale, log=self.log)
        xs.start()
        from Xlib import display as xd
        self.disp = xd.Display()
        t0 = time.time()
        self.step, self.dialog_steps, stuck = 0, 0, 0
        try:
            if not self.launch():
                return
            while self.step < a.steps and time.time() - t0 < a.max_s:
                if self.step % 5 == 0 and self.proc.poll() is not None:
                    import a11y
                    acc = a11y.find_app(APPS[a.app]["atspi"])
                    if acc is None or acc.get_child_count() == 0:
                        self.L("app exited"); break
                    self.app_acc = acc
                try:
                    cap = self.capture()
                except Exception as ex:
                    self.L("capture error", repr(ex)); cap = None
                stuck = stuck + 1 if (cap is None or not cap["saved"]) else 0
                if stuck >= 8:
                    self.L("stuck; escape x3"); sh("xdotool key Escape Escape Escape"); stuck = 0
                self.act(cap)
                self.step += 1
        except Exception:
            self.L(traceback.format_exc())
        finally:
            self.recf.close()
            os.rename(f"{a.out}/jobs/{self.job}.jsonl.tmp", f"{a.out}/jobs/{self.job}.jsonl")
            self.L("done saved", self.n_saved, "steps", self.step, "secs", round(time.time() - t0))
            try:
                kill_tree(self.proc)
            except Exception:
                pass
            xs.stop()
            shutil.rmtree(self.home, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True); ap.add_argument("--display", type=int, required=True)
    ap.add_argument("--W", type=int, default=1920); ap.add_argument("--H", type=int, default=1080)
    ap.add_argument("--scale", type=int, default=1); ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max_s", type=int, default=1200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/data/guisyn"); ap.add_argument("--out", default="/data/guisyn/raw")
    ap.add_argument("--homes", default="/data/guisyn/homes")
    Worker(ap.parse_args()).run()
