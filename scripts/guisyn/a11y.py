"""AT-SPI tree dump + X11 stacking helpers (runs inside a worker's D-Bus session, DISPLAY set).

dump_app(app) -> list of node dicts (full tree, incl. non-showing nodes; extents only for showing nodes):
  {i, p (parent i), d (depth), role, name, desc, states [..], box [x,y,w,h] | None, nchild, actions [..], text}
"""
import os, re, time
import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi

S = Atspi.StateType
KEEP_STATES = {S.SHOWING: "showing", S.VISIBLE: "visible", S.SENSITIVE: "sensitive", S.ENABLED: "enabled",
               S.FOCUSED: "focused", S.SELECTED: "selected", S.CHECKED: "checked", S.PRESSED: "pressed",
               S.EXPANDED: "expanded", S.EXPANDABLE: "expandable", S.EDITABLE: "editable", S.ACTIVE: "active",
               S.MODAL: "modal", S.MANAGES_DESCENDANTS: "manages_desc", S.DEFUNCT: "defunct",
               S.ICONIFIED: "iconified", S.FOCUSABLE: "focusable",
               S.HORIZONTAL: "horizontal", S.VERTICAL: "vertical", S.SELECTABLE: "selectable"}
TEXT_ROLES = {"text", "paragraph", "entry", "label", "heading", "static", "document text", "terminal", "section",
              "table cell", "caption", "link", "list item", "password text", "spin button", "combo box"}
Atspi.init()


def apps():
    d = Atspi.get_desktop(0)
    out = []
    for i in range(d.get_child_count()):
        try:
            a = d.get_child_at_index(i)
            if a is not None:
                out.append(a)
        except Exception:
            pass
    return out


def find_app(pattern):
    """Matching AT-SPI application with the most children (some apps register a helper stub with the same name)."""
    rx = re.compile(pattern, re.I)
    best, bn = None, -1
    for a in apps():
        try:
            if rx.search(a.get_name() or ""):
                n = a.get_child_count()
                if n > bn:
                    best, bn = a, n
        except Exception:
            pass
    return best


def _safe(f, default=None):
    try:
        return f()
    except Exception:
        return default


def dump_app(app, max_nodes=40000, max_children=800, deadline_s=25.0):
    t0 = time.time()
    nodes, objs, stack = [], [], [(app, -1, 0)]
    trunc = 0
    while stack:
        if len(nodes) >= max_nodes or time.time() - t0 > deadline_s:
            trunc += len(stack); break
        acc, parent, depth = stack.pop()
        try:
            role = acc.get_role_name() or ""
            name = acc.get_name() or ""
            ss = acc.get_state_set()
            states = [v for k, v in KEEP_STATES.items() if ss.contains(k)]
            nch = acc.get_child_count()
        except Exception:
            continue
        if "defunct" in states:
            continue
        n = dict(i=len(nodes), p=parent, d=depth, role=role, name=name[:300], states=states, nchild=nch)
        desc = _safe(acc.get_description, "")
        if desc:
            n["desc"] = desc[:300]
        showing = "showing" in states
        if showing or depth <= 1:
            comp = _safe(acc.get_component_iface)
            if comp is not None:
                e = _safe(lambda: comp.get_extents(Atspi.CoordType.SCREEN))
                if e is not None:
                    n["box"] = [e.x, e.y, e.width, e.height]
            act = _safe(acc.get_action_iface)
            if act is not None:
                k = _safe(act.get_n_actions, 0) or 0
                n["actions"] = [_safe(lambda j=j: act.get_action_name(j), "") for j in range(min(k, 4))]
            if role in TEXT_ROLES or not name:
                ti = _safe(acc.get_text_iface)
                if ti is not None:
                    cc = _safe(ti.get_character_count, 0) or 0
                    if cc:
                        n["text"] = _safe(lambda: ti.get_text(0, min(cc, 400)), "")
        if role in ("table", "tree table", "spreadsheet", "list", "tree") and "manages_desc" in states and nch > max_children:
            nch_iter = 0 if not showing else max_children // 4  # huge managed tables (Calc): only the first cells
        else:
            nch_iter = min(nch, max_children)
        if nch > nch_iter:
            n["children_truncated"] = nch - nch_iter
        nodes.append(n)
        objs.append(acc)
        # skip hidden subtrees' children of very deep invisible containers? keep full (needed for negative proofs)
        kids = []
        for j in range(nch_iter):
            c = _safe(lambda j=j: acc.get_child_at_index(j))
            if c is not None:
                kids.append((c, n["i"], depth + 1))
        stack.extend(reversed(kids))
    return nodes, dict(truncated=trunc, secs=round(time.time() - t0, 2), objs=objs)


def _key(acc):
    try:
        e = acc.get_component_iface().get_extents(Atspi.CoordType.SCREEN)
        return (acc.get_role_name(), acc.get_name() or "", e.x, e.y, e.width, e.height)
    except Exception:
        return None


def hit_test(objs, nodes, e, s):
    """Descend get_accessible_at_point from e's top-level at e's box center (logical coords). True if the deepest
    hit is e, one of e's ancestors (toolkit stopped early) or one of e's descendants; False if it lands elsewhere
    (e is hidden / covered by a sibling widget). None if the toolkit gives no answer."""
    by_i = {n["i"]: n for n in nodes}
    n = by_i[e["i"]]
    chain, p = [], n["p"]
    while p >= 0 and p in by_i:
        chain.append(by_i[p]); p = by_i[p]["p"]
    top = next((a for a in chain if a["d"] == 1), None)
    if top is None:
        return None
    b = e["box"]
    x, y = int((b[0] + b[2]) / 2 / s), int((b[1] + b[3]) / 2 / s)
    cur = objs[top["i"]]
    for _ in range(40):
        comp = _safe(cur.get_component_iface)
        if comp is None:
            break
        c = _safe(lambda: comp.get_accessible_at_point(x, y, Atspi.CoordType.SCREEN))
        if c is None or _key(c) == _key(cur):
            break
        cur = c
    k = _key(cur)
    if k is None:
        return None
    nk = lambda m: (m["role"], m["name"], *m["box"]) if m.get("box") else None
    if k == nk(n):
        return "self"
    for j, a_ in enumerate(chain):
        if k == nk(a_):
            return f"anc{j + 1}"
    acc = cur
    for _ in range(12):
        acc = _safe(acc.get_parent)
        if acc is None:
            break
        if _key(acc) == nk(n):
            return "desc"
    return "other"


# ------------------------------------------------------------------ X11 side
def x_stack(disp):
    """Viewable top-level X windows bottom->top: [(xid, (x,y,w,h) outer, (x,y,w,h) client|None, override_redirect, title)]"""
    from Xlib import X
    root = disp.screen().root
    WM_STATE = disp.intern_atom("WM_STATE")
    NAME = disp.intern_atom("_NET_WM_NAME")
    out = []
    for w in root.query_tree().children:
        try:
            a = w.get_attributes()
            if a.map_state != X.IsViewable:
                continue
            g = w.get_geometry()
            outer = (g.x, g.y, g.width, g.height)
            client = None
            title = ""
            def find_client(win, depth=0):
                if win.get_full_property(WM_STATE, X.AnyPropertyType) is not None:
                    return win
                if depth > 3:
                    return None
                for c in win.query_tree().children:
                    r = find_client(c, depth + 1)
                    if r is not None:
                        return r
                return None
            cw = find_client(w)
            if cw is not None:
                cg = cw.get_geometry()
                tr = cw.translate_coords(root, 0, 0)
                client = (-tr.x, -tr.y, cg.width, cg.height)
                p = cw.get_full_property(NAME, 0)
                title = p.value.decode("utf8", "ignore") if p is not None and isinstance(p.value, bytes) else ""
            out.append((w.id, outer, client, bool(a.override_redirect), title))
        except Exception:
            continue
    return out


def grab(disp, W, H):
    from Xlib import X
    from PIL import Image
    raw = disp.screen().root.get_image(0, 0, W, H, X.ZPixmap, 0xffffffff)
    return Image.frombytes("RGB", (W, H), raw.data, "raw", "BGRX")
