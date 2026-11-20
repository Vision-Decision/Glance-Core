"""Element filtering for one desktop screenshot: AT-SPI nodes + X stacking -> visible target elements (physical px).

Rules (all must hold):
  - node SHOWING (+VISIBLE), has extents, not DEFUNCT, role in TARGET_ROLES;
  - scaled to physical px (scale calibrated from top-level frames vs X client windows), clipped to the screen, to
    every scrolling ancestor (viewport / scroll pane) and to its top-level window; >= 90% of the box must survive;
  - min size 6x6 px (x scale), max 35% of the screen;
  - not occluded: on a 5x5 grid inside the box, the topmost X window at >= 22/25 points is the element's own
    top-level window, or (for elements inside a menu / combo popup subtree) an override-redirect popup window;
  - pixel content is not flat (std > 2) - drops elements reported as showing but not painted;
  - name non-empty, unless icon-like (small, roughly square, clickable) -> kept as `uncaptioned` (no question);
  - near-duplicate boxes (IoU > 0.85) collapse to one element (prefer interactive role, then non-empty name).
"""
import re
import numpy as np

TOPLEVEL_ROLES = {"frame", "dialog", "window", "file chooser", "alert", "color chooser", "font chooser", "menu",
                  "popup menu", "tool tip", "notification"}
INTERACTIVE = {"push button", "toggle button", "check box", "radio button", "menu item", "check menu item",
               "radio menu item", "menu", "page tab", "combo box", "spin button", "slider", "link", "entry", "text",
               "password text", "list item", "tree item", "table cell", "icon", "button", "split button",
               "toggle", "scroll bar", "tool bar item", "push button menu", "dial", "color chooser", "date editor",
               "table column header", "column header", "row header", "editbar", "terminal"}
TARGET_ROLES = (INTERACTIVE - {"scroll bar", "terminal", "editbar"}) | {"label", "heading", "image", "static",
                                                                      "caption", "paragraph", "tool tip", "progress bar"}
CONTAINER_ROLES = {"tool bar", "menu bar", "page tab list", "list", "tree", "tree table", "table", "panel", "group",
                   "status bar", "section", "landmark", "filler", "split pane", "scroll pane", "viewport"}
CLIP_ROLES = {"viewport", "scroll pane"}
POPUP_ANCESTOR = {"menu", "popup menu", "combo box", "list box"}
ICONLIKE_ROLES = {"push button", "toggle button", "icon", "button", "radio button", "split button", "tool bar item",
                  "image", "menu", "push button menu", "check box", "page tab", "link"}


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _inter(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def xyxy(box, s=1.0, off=(0, 0)):
    x, y, w, h = box
    return [x * s + off[0], y * s + off[1], (x + w) * s + off[0], (y + h) * s + off[1]]


def calibrate(nodes, xstack):
    """Match depth-1 showing top-levels to X client windows. Returns (scale, {node_i: x_index}, matches info)."""
    clients = [(k, c) for k, (_, o, c, orr, _) in enumerate(xstack) if c is not None]
    tops = [n for n in nodes if n["d"] == 1 and n.get("box") and n["box"][2] > 50 and n["box"][3] > 50]
    best_s, best = 1.0, (-1, {})
    for s in (1.0, 2.0, 1.5, 1.25):
        m = {}
        for n in tops:
            b = xyxy(n["box"], s)
            cand = [(_iou(b, xyxy(c)), k) for k, c in clients]
            if cand:
                v, k = max(cand)
                if v > 0.8:
                    m[n["i"]] = k
        if len(m) > best[0]:
            best_s, best = s, (len(m), m)
    return best_s, best[1]


def stacking_top(xstack, W, H, step=4):
    """Low-res map: index (into xstack) of topmost viewable window per (step x step) cell; -1 = root."""
    h, w = (H + step - 1) // step, (W + step - 1) // step
    top = np.full((h, w), -1, np.int32)
    for k, (_, o, c, orr, _) in enumerate(xstack):  # bottom -> top, outer rect (incl. decorations)
        x, y, ww, hh = o
        x0, y0 = max(0, x // step), max(0, y // step)
        x1, y1 = min(w, (x + ww + step - 1) // step), min(h, (y + hh + step - 1) // step)
        if x1 > x0 and y1 > y0:
            top[y0:y1, x0:x1] = k
    return top


def filter_elements(nodes, xstack, W, H, img=None, scale=None, keep_debug=False):
    s, tmatch = calibrate(nodes, xstack)
    if scale:
        s = scale
    by_i = {n["i"]: n for n in nodes}
    top = stacking_top(xstack, W, H)
    step = 4
    override = {k for k, x in enumerate(xstack) if x[3]}
    gray = np.asarray(img.convert("L"), dtype=np.float32) if img is not None else None
    stats = dict(nodes=len(nodes), showing=0, cand=0, occluded=0, clipped=0, small=0, flat=0, big=0, unmatched_top=0,
                 kept=0, uncaptioned=0, dup=0)
    out = []
    for n in nodes:
        st = n["states"]
        if "showing" not in st or not n.get("box"):
            continue
        stats["showing"] += 1
        role = n["role"]
        if role not in TARGET_ROLES:
            continue
        stats["cand"] += 1
        # ancestors
        chain, p = [], n["p"]
        while p >= 0 and p in by_i:
            chain.append(by_i[p]); p = by_i[p]["p"]
        if any("showing" not in a["states"] for a in chain if a["d"] >= 1):
            stats["anc_hidden"] = stats.get("anc_hidden", 0) + 1; continue
        topn = next((a for a in chain if a["d"] == 1), n if n["d"] == 1 else None)
        if topn is None or topn["i"] not in tmatch:
            stats["unmatched_top"] += 1; continue
        xk = tmatch[topn["i"]]
        b = xyxy(n["box"], s)
        a0 = _area(b)
        if a0 <= 0:
            continue
        clip = [0, 0, W, H]
        c = xstack[xk][2]
        clip = _inter(clip, [c[0], c[1], c[0] + c[2], c[1] + c[3]])
        in_popup = any(a["role"] in POPUP_ANCESTOR for a in chain[:6])
        for a in chain:
            if a["role"] in CLIP_ROLES and a.get("box") and not in_popup:
                clip = _inter(clip, xyxy(a["box"], s))
        if in_popup:
            clip = [0, 0, W, H]
        bc = _inter(b, clip)
        if _area(bc) < 0.9 * a0:
            stats["clipped"] += 1; continue
        b = bc
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw < 6 * s or bh < 6 * s:
            stats["small"] += 1; continue
        if bw * bh > 0.35 * W * H:
            stats["big"] += 1; continue
        # occlusion grid
        xs = np.linspace(b[0] + 1, b[2] - 1, 5); ys = np.linspace(b[1] + 1, b[3] - 1, 5)
        ok = 0
        for yy in ys:
            for xx in xs:
                t = top[min(top.shape[0] - 1, int(yy) // step), min(top.shape[1] - 1, int(xx) // step)]
                if t == xk or (in_popup and t in override and t > xk):
                    ok += 1
        if ok < 22:
            stats["occluded"] += 1; continue
        if gray is not None:
            reg = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            if reg.size == 0 or float(reg.std()) < 2.0:
                stats["flat"] += 1; continue
        name = re.sub(r"\s+", " ", n.get("name") or "").strip()
        iconlike = role in ICONLIKE_ROLES and bw <= 64 * s and bh <= 64 * s and 0.5 <= bw / max(bh, 1) <= 2.0
        if in_popup:
            pass
        e = dict(i=n["i"], role=role, name=name, box=[round(v, 1) for v in b], popup=in_popup,
                 states=[x for x in st if x in ("checked", "selected", "pressed", "expanded", "focused", "sensitive", "editable")],
                 icon=bool(iconlike), depth=n["d"])
        if n.get("desc"):
            e["desc"] = re.sub(r"\s+", " ", n["desc"]).strip()
        if n.get("text") and role in ("text", "entry", "paragraph", "label", "static", "heading", "table cell",
                                       "combo box", "spin button", "list item"):
            e["text"] = re.sub(r"\s+", " ", n["text"]).strip()[:200]
        # context: named ancestors, top-level, menu path
        e["ctx"] = context(n, chain)
        if not name:
            if not iconlike or role in ("image",):
                if role in ("text", "entry", "paragraph") and e.get("text"):
                    pass
                else:
                    continue
            else:
                e["uncaptioned"] = True
                stats["uncaptioned"] += 1
        out.append(e)
    # dedup near-identical boxes
    pri = lambda e: (e["role"] in INTERACTIVE, bool(e["name"]), -e["depth"])
    out.sort(key=pri, reverse=True)
    kept = []
    for e in out:
        if any(_iou(e["box"], k["box"]) > 0.85 for k in kept):
            stats["dup"] += 1; continue
        kept.append(e)
    stats["kept"] = len(kept)
    stats["scale"] = s
    return kept, stats


def context(n, chain):
    ctx = {}
    names = []
    for a in chain:
        r, nm = a["role"], (a.get("name") or "").strip()
        if a["d"] == 1:
            ctx["top"] = dict(role=r, name=nm[:80])
            break
        if r in ("menu",) and nm:
            ctx.setdefault("menu_path", []).insert(0, nm)
        elif r == "tool bar":
            ctx.setdefault("toolbar", nm or "")
            if a.get("box"):
                ctx.setdefault("toolbar_box", a["box"])
        elif r == "page tab" and nm:
            ctx.setdefault("tab", nm)
        elif r in ("dialog", "panel", "group", "section", "list", "tree", "tree table", "table", "page tab list",
                   "menu bar", "status bar", "landmark", "tool bar", "scroll pane", "filler") and nm:
            names.append((r, nm[:60]))
    if names:
        ctx["named_anc"] = names[:3]
    ctx["parent_role"] = chain[0]["role"] if chain else ""
    ctx["parent_i"] = chain[0]["i"] if chain else -1
    return ctx
