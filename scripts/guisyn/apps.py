"""Desktop app registry. Each entry: launch command (with {doc}), AT-SPI app-name regex, sample-doc kind, toolkit."""
import glob, os, random

SAMPLES = "/data/guisyn/samples"


def docs(kind):
    return sorted(glob.glob(f"{SAMPLES}/{kind}/*"))


APPS = {
    # name: dict(cmd, atspi, docs (kind or callable), tk, wait)
    "lo_writer": dict(cmd="soffice --nologo --norestore --writer {doc}", atspi=r"soffice|libreoffice", docs="docx", tk="gtk3", wait=12,
                      env=dict(SAL_USE_VCLPLUGIN="gtk3")),
    "lo_calc": dict(cmd="soffice --nologo --norestore --calc {doc}", atspi=r"soffice|libreoffice", docs="xlsx", tk="gtk3", wait=12,
                    env=dict(SAL_USE_VCLPLUGIN="gtk3")),
    "lo_impress": dict(cmd="soffice --nologo --norestore --impress {doc}", atspi=r"soffice|libreoffice", docs="pptx", tk="gtk3", wait=12,
                       env=dict(SAL_USE_VCLPLUGIN="gtk3")),
    "lo_draw": dict(cmd="soffice --nologo --norestore --draw {doc}", atspi=r"soffice|libreoffice", docs="svg", tk="gtk3", wait=12,
                    env=dict(SAL_USE_VCLPLUGIN="gtk3")),
    "gimp": dict(cmd="gimp --no-splash {doc}", atspi=r"gimp", docs="img", tk="gtk2", wait=25),
    "inkscape": dict(cmd="inkscape {doc}", atspi=r"inkscape", docs="svg", tk="gtk3", wait=15),
    "krita": dict(cmd="krita --nosplash {doc}", atspi=r"krita", docs="img", tk="qt5", wait=25),
    "freecad": dict(cmd="freecad {doc}", atspi=r"freecad", docs="fcstd", tk="qt5", wait=20),
    "kicad_pcb": dict(cmd="pcbnew {doc}", atspi=r"pcbnew|kicad", docs="kicad_pcb", tk="wx", wait=15),
    "kicad_sch": dict(cmd="eeschema {doc}", atspi=r"eeschema|kicad", docs="kicad_sch", tk="wx", wait=15),
    "codium": dict(cmd="/usr/share/codium/codium --no-sandbox --disable-gpu --force-renderer-accessibility --disable-workspace-trust "
                       "--user-data-dir {home}/.codium --extensions-dir {home}/.codium-ext {doc}",
                   atspi=r"codium|code|electron", docs="code", tk="electron", wait=15),
    "firefox": dict(cmd="firefox --new-instance --profile {home}/.ffprofile {doc}", atspi=r"firefox", docs="ffurl", tk="gtk3", wait=12),
    "thunderbird": dict(cmd="thunderbird --new-instance --profile {home}/.tbprofile", atspi=r"thunderbird", docs=None, tk="gtk3", wait=15),
    "kdenlive": dict(cmd="kdenlive {doc}", atspi=r"kdenlive", docs="video", tk="qt5", wait=20),
    "shotcut": dict(cmd="shotcut {doc}", atspi=r"shotcut", docs="video", tk="qt5", wait=15),
    "audacity": dict(cmd="audacity {doc}", atspi=r"audacity", docs="audio", tk="wx", wait=15),
    "darktable": dict(cmd="darktable --configdir {home}/.dt {doc}", atspi=r"darktable", docs="img", tk="gtk3", wait=15),
    "qgis": dict(cmd="qgis --nologo --noversioncheck --noplugins {doc}", atspi=r"qgis", docs="geo", tk="qt5", wait=25),
    "wireshark": dict(cmd="wireshark -r {doc}", atspi=r"wireshark", docs="pcap", tk="qt5", wait=12),
}

# app-specific keyboard shortcuts that open panels / dialogs / views (never quit/save/print)
KEYS = {
    "lo_writer": ["F11", "ctrl+F8", "F5", "ctrl+shift+F10", "ctrl+h", "F7"],
    "lo_calc": ["F11", "ctrl+F2", "ctrl+1", "ctrl+h", "F5"],
    "lo_impress": ["F11", "ctrl+F2", "F5"],
    "lo_draw": ["F11", "F4", "F5"],
    "gimp": ["ctrl+b", "Tab", "ctrl+l", "ctrl+shift+j", "plus", "minus", "1"],
    "inkscape": ["ctrl+shift+f", "ctrl+shift+l", "ctrl+shift+a", "ctrl+shift+t", "ctrl+shift+x", "5", "3", "F1", "F2", "F3", "F4"],
    "krita": ["Tab", "F5", "plus", "minus", "ctrl+shift+a"],
    "freecad": ["0", "1", "2", "3", "v", "ctrl+shift+F12"],
    "kicad_pcb": ["alt+3", "ctrl+shift+F", "Home", "F1", "F2"],
    "kicad_sch": ["Home", "F1", "F2", "ctrl+f"],
    "codium": ["ctrl+shift+e", "ctrl+shift+f", "ctrl+shift+x", "ctrl+shift+d", "ctrl+j", "ctrl+b", "ctrl+comma", "ctrl+shift+m", "ctrl+p"],
    "firefox": ["ctrl+shift+b", "ctrl+h", "ctrl+f", "ctrl+plus", "ctrl+minus", "ctrl+0"],
    "thunderbird": ["F8", "ctrl+shift+b", "ctrl+f"],
    "kdenlive": ["ctrl+shift+comma"],
    "shotcut": ["ctrl+comma"],
    "audacity": ["ctrl+1", "ctrl+3", "ctrl+a", "ctrl+p"],
    "darktable": ["d", "l", "Tab", "ctrl+shift+h"],
    "qgis": ["ctrl+shift+F", "ctrl+1", "ctrl+2", "ctrl+3", "ctrl+5"],
    "wireshark": ["ctrl+right", "ctrl+down", "ctrl+slash", "ctrl+alt+shift+i", "ctrl+shift+e"],
}


def pick_doc(app, rng):
    k = APPS[app]["docs"]
    if not k:
        return ""
    ds = docs(k)
    return rng.choice(ds) if ds else ""
