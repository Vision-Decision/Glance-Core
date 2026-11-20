"""X session for one worker: Xvfb (no TCP) + at-spi bus + openbox. Must run inside `dbus-run-session`."""
import os, shutil, signal, subprocess, time

THEMES_GTK = ["Adwaita", "Adwaita-dark", "Yaru", "Yaru-dark", "Arc", "Arc-Dark", "HighContrast"]
ICONS = ["Adwaita", "Yaru", "Papirus", "Papirus-Dark", "breeze"]
QT_STYLES = ["Fusion", "Breeze", "Windows", "kvantum"]


class XSession:
    def __init__(self, display, W, H, scale=1, log=None):
        self.display, self.W, self.H, self.scale = display, W, H, scale
        self.procs = []
        self.log = log or open(os.devnull, "w")

    def start(self):
        d = f":{self.display}"
        lock = f"/tmp/.X{self.display}-lock"
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except OSError:
                pass
        self.procs.append(subprocess.Popen(
            ["Xvfb", d, "-screen", "0", f"{self.W}x{self.H}x24", "-nolisten", "tcp", "-dpi", str(96 * self.scale),
             "-ac", "+extension", "RANDR", "-noreset"], stdout=self.log, stderr=self.log))
        os.environ["DISPLAY"] = d
        for _ in range(100):
            if os.path.exists(f"/tmp/.X11-unix/X{self.display}"):
                break
            time.sleep(0.1)
        time.sleep(0.5)
        env = os.environ
        env.update(NO_AT_BRIDGE="0", GTK_MODULES="gail:atk-bridge", GNOME_ACCESSIBILITY="1",
                   QT_ACCESSIBILITY="1", QT_LINUX_ACCESSIBILITY_ALWAYS_ON="1", ACCESSIBILITY_ENABLED="1",
                   XDG_SESSION_TYPE="x11", GDK_BACKEND="x11", QT_QPA_PLATFORM="xcb", NO_AT_BRIDGE_FORCE="0",
                   BROWSER="/bin/true", MOZ_ENABLE_WAYLAND="0", LIBGL_ALWAYS_SOFTWARE="1",
                   XDG_CURRENT_DESKTOP="GNOME")
        self.procs.append(subprocess.Popen(["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"],
                                           stdout=self.log, stderr=self.log))
        time.sleep(0.8)
        subprocess.run(["gsettings", "set", "org.gnome.desktop.interface", "toolkit-accessibility", "true"],
                       stdout=self.log, stderr=self.log)
        self.procs.append(subprocess.Popen(["openbox", "--sm-disable"], stdout=self.log, stderr=self.log))
        time.sleep(0.5)
        subprocess.run(["xsetroot", "-solid", "#3a6ea5"], stdout=self.log, stderr=self.log)

    def stop(self):
        for p in reversed(self.procs):
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(0.3)
        for p in self.procs:
            try:
                p.kill()
            except Exception:
                pass


def kill_tree(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(1.0)
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        pass
