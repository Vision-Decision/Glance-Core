"""Generate synthetic sample documents that the desktop apps open (all generated locally; no third-party content).

  python make_samples.py --out /data/guisyn/samples
kinds: docx xlsx pptx svg img video audio pcap geo code ffurl kicad_pcb kicad_sch fcstd
"""
import argparse, glob, json, math, os, random, shutil, subprocess

WORDS = ("project budget review quarterly report analysis design system model network sample data value total "
         "revenue market customer product service team meeting schedule summary result method figure table "
         "chapter section introduction conclusion appendix reference version update policy process quality "
         "research energy climate water transport city region county population growth survey index rate "
         "circuit layer signal frequency amplitude filter render texture vector shape color palette gradient "
         "inventory order invoice shipment supplier warehouse account balance forecast target milestone").split()
NAMES = ["Alice Chen", "Bob Martin", "Carla Diaz", "Deepak Rao", "Eva Novak", "Farid Haddad", "Grace Kim", "Hugo Laurent"]


def sent(r, n=None):
    n = n or r.randint(6, 16)
    s = " ".join(r.choice(WORDS) for _ in range(n))
    return s[0].upper() + s[1:] + "."


def docx(out, r, n):
    from docx import Document
    for k in range(n):
        d = Document()
        d.add_heading(f"{r.choice(WORDS).title()} {r.choice(WORDS).title()} {r.choice(['Report', 'Notes', 'Plan', 'Memo'])}", 0)
        for _ in range(r.randint(3, 8)):
            if r.random() < 0.3:
                d.add_heading(sent(r, 3)[:-1], r.randint(1, 2))
            d.add_paragraph(" ".join(sent(r) for _ in range(r.randint(2, 6))))
            if r.random() < 0.25:
                t = d.add_table(rows=r.randint(2, 5), cols=r.randint(2, 4)); t.style = "Table Grid"
                for row in t.rows:
                    for c in row.cells:
                        c.text = str(r.randint(1, 999)) if r.random() < 0.5 else r.choice(WORDS)
            if r.random() < 0.3:
                for _ in range(r.randint(2, 4)):
                    d.add_paragraph(sent(r, 5), style="List Bullet")
        d.save(f"{out}/doc_{k:03d}.docx")


def xlsx(out, r, n):
    from openpyxl import Workbook
    for k in range(n):
        wb = Workbook()
        for si in range(r.randint(1, 3)):
            ws = wb.active if si == 0 else wb.create_sheet()
            ws.title = r.choice(WORDS).title() + str(si)
            cols = r.randint(3, 10)
            ws.append([r.choice(WORDS).title() for _ in range(cols)])
            for i in range(r.randint(10, 60)):
                ws.append([r.choice(NAMES) if c == 0 else round(r.uniform(0, 1000), 2) for c in range(cols)])
            if r.random() < 0.5:
                ws.append(["Total"] + [f"=SUM({chr(66 + c)}2:{chr(66 + c)}{ws.max_row})" for c in range(cols - 1)])
        wb.save(f"{out}/sheet_{k:03d}.xlsx")


def pptx(out, r, n):
    from pptx import Presentation
    from pptx.util import Inches
    for k in range(n):
        p = Presentation()
        for s in range(r.randint(3, 8)):
            sl = p.slides.add_slide(p.slide_layouts[r.choice([0, 1, 1, 5])])
            if sl.shapes.title is not None:
                sl.shapes.title.text = sent(r, 4)[:-1]
            for ph in sl.placeholders:
                if ph.placeholder_format.idx == 1:
                    ph.text = "\n".join(sent(r, 6) for _ in range(r.randint(2, 5)))
        p.save(f"{out}/deck_{k:03d}.pptx")


def svg(out, r, n):
    for k in range(n):
        W, H = r.choice([(800, 600), (1200, 800), (600, 600)])
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
        for i in range(r.randint(5, 30)):
            col = "#%06x" % r.randint(0, 0xFFFFFF)
            t = r.random()
            if t < 0.3:
                parts.append(f'<rect id="rect{i}" x="{r.randint(0, W - 50)}" y="{r.randint(0, H - 50)}" width="{r.randint(20, 300)}" '
                             f'height="{r.randint(20, 200)}" fill="{col}" stroke="#222" stroke-width="{r.randint(0, 4)}"/>')
            elif t < 0.6:
                parts.append(f'<circle id="circle{i}" cx="{r.randint(0, W)}" cy="{r.randint(0, H)}" r="{r.randint(10, 120)}" fill="{col}" opacity="{r.uniform(0.4, 1):.2f}"/>')
            elif t < 0.8:
                pts = " ".join(f"{r.randint(0, W)},{r.randint(0, H)}" for _ in range(r.randint(3, 7)))
                parts.append(f'<polygon id="poly{i}" points="{pts}" fill="{col}"/>')
            else:
                parts.append(f'<text id="text{i}" x="{r.randint(0, W - 200)}" y="{r.randint(30, H)}" font-size="{r.randint(14, 48)}" '
                             f'font-family="DejaVu Sans" fill="{col}">{sent(r, 3)[:-1]}</text>')
        parts.append("</svg>")
        open(f"{out}/draw_{k:03d}.svg", "w").write("\n".join(parts))


def img(out, r, n):
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    for k in range(n):
        W, H = r.choice([(1600, 1200), (2400, 1600), (1024, 768), (3000, 2000)])
        y, x = np.mgrid[0:H, 0:W] / max(W, H)
        c = [np.sin(x * r.uniform(1, 9) + r.uniform(0, 6)) * np.cos(y * r.uniform(1, 9)) for _ in range(3)]
        a = np.stack([(127 + 127 * ch) for ch in c], -1).astype(np.uint8)
        im = Image.fromarray(a)
        d = ImageDraw.Draw(im)
        for _ in range(r.randint(5, 40)):
            col = tuple(r.randint(0, 255) for _ in range(3))
            x0, y0 = r.randint(0, W), r.randint(0, H)
            if r.random() < 0.5:
                d.ellipse([x0, y0, x0 + r.randint(20, 500), y0 + r.randint(20, 500)], fill=col)
            else:
                d.rectangle([x0, y0, x0 + r.randint(20, 500), y0 + r.randint(20, 500)], fill=col)
        im = im.filter(ImageFilter.GaussianBlur(r.uniform(0, 3)))
        ext = r.choice(["png", "jpg", "jpg"])
        im.save(f"{out}/img_{k:03d}.{ext}", quality=92)


def video(out, r, n):
    srcs = ["testsrc2", "mandelbrot", "life", "smptehdbars", "cellauto", "rgbtestsrc"]
    for k in range(n):
        s = r.choice(srcs)
        size = r.choice(["1280x720", "1920x1080", "640x480"])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"{s}=size={size}:rate=25",
                        "-f", "lavfi", "-i", f"sine=frequency={r.randint(200, 900)}", "-t", str(r.randint(4, 15)),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", f"{out}/clip_{k:03d}.mp4"])


def audio(out, r, n):
    import numpy as np, wave
    for k in range(n):
        sr = 44100; T = r.uniform(3, 20)
        t = np.arange(int(sr * T)) / sr
        x = sum(r.uniform(0.1, 0.4) * np.sin(2 * np.pi * r.uniform(80, 2000) * t * (1 + r.uniform(0, 0.5) * t / T)) for _ in range(3))
        x = x * (0.5 + 0.5 * np.sin(2 * np.pi * r.uniform(0.2, 2) * t)) + r.uniform(0, 0.05) * np.random.randn(len(t))
        x = np.clip(x, -1, 1)
        ch = r.choice([1, 2])
        data = np.repeat((x * 30000).astype(np.int16)[:, None], ch, 1)
        w = wave.open(f"{out}/tone_{k:03d}.wav", "wb"); w.setnchannels(ch); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes()); w.close()


def pcap(out, r, n):
    from scapy.all import IP, TCP, UDP, DNS, DNSQR, Ether, Raw, ICMP, wrpcap
    hosts = [f"10.0.{r.randint(0, 9)}.{r.randint(2, 250)}" for _ in range(12)]
    for k in range(n):
        pk = []
        for i in range(r.randint(80, 600)):
            s, d = r.sample(hosts, 2)
            t = r.random()
            if t < 0.3:
                p = Ether() / IP(src=s, dst=d) / UDP(sport=r.randint(1024, 65000), dport=53) / DNS(rd=1, qd=DNSQR(qname=f"{r.choice(WORDS)}.example.org"))
            elif t < 0.75:
                p = Ether() / IP(src=s, dst=d) / TCP(sport=r.randint(1024, 65000), dport=r.choice([80, 443, 22, 8080]), flags=r.choice(["S", "SA", "A", "PA", "F"])) / Raw(
                    (f"GET /{r.choice(WORDS)} HTTP/1.1\r\nHost: {r.choice(WORDS)}.example.org\r\n\r\n" if r.random() < 0.3 else "x" * r.randint(0, 200)).encode())
            else:
                p = Ether() / IP(src=s, dst=d) / ICMP()
            p.time = 1700000000 + i * r.uniform(0.001, 0.2)
            pk.append(p)
        wrpcap(f"{out}/trace_{k:03d}.pcap", pk)


def geo(out, r, n):
    for k in range(n):
        lon0, lat0 = r.uniform(-120, 120), r.uniform(-50, 60)
        feats = []
        for i in range(r.randint(5, 60)):
            if r.random() < 0.5:
                cx, cy = lon0 + r.uniform(-5, 5), lat0 + r.uniform(-5, 5)
                ring = [[cx + r.uniform(0.2, 1.2) * math.cos(a), cy + r.uniform(0.2, 1.2) * math.sin(a)]
                        for a in [j * 2 * math.pi / 8 for j in range(8)]]
                ring.append(ring[0])
                g = dict(type="Polygon", coordinates=[ring])
            else:
                g = dict(type="Point", coordinates=[lon0 + r.uniform(-5, 5), lat0 + r.uniform(-5, 5)])
            feats.append(dict(type="Feature", geometry=g, properties=dict(name=f"{r.choice(WORDS).title()} {i}",
                                                                          value=r.randint(0, 1000), kind=r.choice(WORDS))))
        json.dump(dict(type="FeatureCollection", features=feats), open(f"{out}/layer_{k:03d}.geojson", "w"))


def code(out, r, n):
    import sysconfig
    lib = sysconfig.get_paths()["stdlib"]
    pys = sorted(glob.glob(f"{lib}/*.py"))
    for k in range(n):
        d = f"{out}/proj_{k:03d}"
        os.makedirs(f"{d}/src", exist_ok=True)
        for f in r.sample(pys, r.randint(3, 10)):
            shutil.copy(f, f"{d}/src/")
        open(f"{d}/README.md", "w").write(f"# {r.choice(WORDS).title()} {r.choice(WORDS)}\n\n{sent(r)}\n")


def ffurl(out, r, n):
    urls = ["about:preferences", "about:preferences#privacy", "about:preferences#search", "about:addons", "about:support",
            "about:config", "about:downloads", "about:profiles", "about:logins", "about:newtab", "about:processes",
            "about:preferences#home", "about:policies", "about:license"]
    for i, u in enumerate(urls):
        open(f"{out}/u{i:02d}.url", "w").write(u)
    # local HTML pages generated from the doc samples (no network)
    for k in range(n):
        body = "".join(f"<h2>{sent(r, 3)}</h2><p>{' '.join(sent(r) for _ in range(5))}</p>" +
                       (f"<form><label>{r.choice(WORDS)} <input name=q></label><button type=button>{r.choice(WORDS).title()}</button>"
                        f"<select><option>{r.choice(WORDS)}</option><option>{r.choice(WORDS)}</option></select></form>" if r.random() < 0.5 else "")
                       for _ in range(r.randint(2, 6)))
        p = f"{out}/page_{k:03d}.html"
        open(p, "w").write(f"<!doctype html><title>{sent(r, 3)}</title><body style='font-family:sans-serif;max-width:900px;margin:auto'>{body}</body>")


def kicad(out, r):
    for ext, kind in [("kicad_pcb", "kicad_pcb"), ("kicad_sch", "kicad_sch")]:
        os.makedirs(f"{out}/{kind}", exist_ok=True)
        for f in glob.glob(f"/usr/share/kicad/demos/**/*.{ext}", recursive=True):
            if os.path.getsize(f) < 20_000_000:
                dst = f"{out}/{kind}/{os.path.basename(os.path.dirname(f))}__{os.path.basename(f)}"
                shutil.copy(f, dst)


def fcstd(out):
    os.makedirs(f"{out}/fcstd", exist_ok=True)
    for f in glob.glob("/usr/share/freecad*/examples/*.FCStd") + glob.glob("/usr/share/freecad*/**/*.FCStd", recursive=True):
        shutil.copy(f, f"{out}/fcstd/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/guisyn/samples")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    r = random.Random(0)
    for kind, fn in [("docx", docx), ("xlsx", xlsx), ("pptx", pptx), ("svg", svg), ("img", img), ("video", video),
                     ("audio", audio), ("pcap", pcap), ("geo", geo), ("code", code), ("ffurl", ffurl)]:
        os.makedirs(f"{a.out}/{kind}", exist_ok=True)
        try:
            fn(f"{a.out}/{kind}", r, a.n if kind not in ("video",) else 12)
            print(kind, len(os.listdir(f"{a.out}/{kind}")))
        except Exception as e:
            print("FAILED", kind, repr(e))
    kicad(a.out, r); fcstd(a.out)
    for k in ("kicad_pcb", "kicad_sch", "fcstd"):
        print(k, len(os.listdir(f"{a.out}/{k}")))
