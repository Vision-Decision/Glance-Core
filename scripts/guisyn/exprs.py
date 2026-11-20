"""Template referring expressions for desktop/web elements + absence verification for negatives."""
import re
from collections import Counter

NOUN = {"push button": "button", "button": "button", "toggle button": "toggle button", "check box": "checkbox",
        "radio button": "radio button", "menu item": "menu item", "check menu item": "menu item",
        "radio menu item": "menu item", "menu": "menu", "page tab": "tab", "combo box": "dropdown", "entry": "text field",
        "text": "text field", "password text": "password field", "link": "link", "list item": "list item",
        "tree item": "tree item", "label": "label", "heading": "heading", "table cell": "cell", "slider": "slider",
        "spin button": "spin box", "icon": "icon", "image": "image", "tool bar": "toolbar", "status bar": "status bar",
        "table column header": "column header", "column header": "column header", "split button": "button",
        "push button menu": "menu button", "tool bar item": "toolbar item", "canvas": "canvas", "paragraph": "paragraph",
        "page tab list": "tab bar", "menu bar": "menu bar", "panel": "panel", "progress bar": "progress bar",
        "static": "text", "caption": "caption", "tree table": "table", "table": "table", "list": "list", "tree": "tree",
        "group": "group", "section": "section", "landmark": "region", "row header": "row header", "dial": "dial"}
CHROME_ROLES = {"push button", "toggle button", "check box", "radio button", "menu item", "check menu item",
                "radio menu item", "menu", "page tab", "combo box", "link", "tree item", "list item", "label", "heading",
                "slider", "spin button", "icon", "split button", "button", "push button menu", "tool bar item",
                "column header", "table column header", "entry"}
WEB_REGIONS = {"navigation", "header", "footer", "sidebar", "form", "dialog", "search", "region"}
ICON_KIND_ROLES = {"push button", "toggle button", "icon", "button", "split button", "tool bar item", "push button menu",
                   "radio button", "image"}
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
DOC_WORDS = set(("project budget review quarterly report analysis design system model network sample data value total "
                 "revenue market customer product service team meeting schedule summary result method figure table "
                 "chapter section introduction conclusion appendix reference version update policy process quality "
                 "research energy climate water transport city region county population growth survey index rate "
                 "circuit layer signal frequency amplitude filter render texture vector shape color palette gradient "
                 "inventory order invoice shipment supplier warehouse account balance forecast target milestone").split())


def norm(s):
    return re.sub(r"[\W_]+", " ", (s or "").lower()).strip()


SHORTCUT = re.compile(r"\s*[\(\[]\s*(ctrl|alt|shift|meta|super|cmd|f\d{1,2}\b)[^\)\]]*[\)\]]\s*$", re.I)


ICON_ID = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)+$|-symbolic$|^gtk-|^qrc:|\.(svg|png)$")
DOUBLED = re.compile(r"^(.{2,60}?)\s+\1$")
BARE_KEY = re.compile(r"(\s+(ctrl|alt|shift|meta|super|cmd)\+\S+)+$", re.I)  # "Pin Ctrl+K Shift+Enter"
TRAIL_KEY = re.compile(r"\s+\((?:[A-Z0-9][\w+\-\]\[ ]{0,11}|[^()\s]{1,3})\)$")  # "(H)", "(3)", "(Page Down)", "(Ctrl+])"
TAGS = re.compile(r"<[^<>]{1,300}>")
RICH = re.compile(r"<\w+\s+[^<>]*=")  # rich-text tags with attributes (FreeCAD tooltips: "<p style=..>Menu<p ..>Std_Cmd")
PUA = re.compile(r"[\ue000-\uf8ff\U000f0000-\U0010ffff]")


def clean(name):
    name = PUA.sub("", TAGS.sub("", name or ""))
    name = SHORTCUT.sub("", re.sub(r"\s+", " ", name)).strip()
    name = BARE_KEY.sub("", name).strip() or name
    m = DOUBLED.match(name)
    if m:
        name = m.group(1)  # "Products Products" (web innerText + aria duplicates)
    return TRAIL_KEY.sub("", name).strip() if len(name) > 12 else name


def good_name(name):
    """>= 2 letters, not a bare number / size / id like '693' or '12 px'."""
    if ICON_ID.match(name) or "_" in name and " " not in name and name.lower() == name:
        return False  # icon-theme ids ("edit-clear-symbolic", "go-next") / internal ids ("std_viewleft")
    letters = sum(ch.isalpha() for ch in name)
    if letters < 2 or len(name) > 120:
        return False
    if re.fullmatch(r"[\d\s.,:%x×+\-]*(px|pt|mm|cm|in|%|kb|mb|kib|mib)?", name.lower()):
        return False
    return True


def good_desc(d, name):
    if RICH.search(d or ""):
        return ""
    d = clean(d)
    if not d or " — " in d or " - " in d or len(d.split()) < 2 or len(d) > 160:
        return ""
    if norm(d) == norm(name) or norm(d).startswith(norm(name) + " ") and len(d) < len(name) + 6:
        return ""
    return d


def q(name):
    name = name.strip()
    return f'"{name}"' if len(name) <= 60 else f'"{name[:57]}..."'


def cell9(b, W, H):
    cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
    v = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    return "center" if (v, h) == ("middle", "center") else (f"{v}-{h}" if v != "middle" else f"{h} side" if h != "center" else "center")


def element_exprs(e, rec_els, W, H, rng):
    """-> list of (text, kind, meta) for one visible element (possibly empty if ambiguous)."""
    out = []
    name = clean(e["name"])
    role = e["role"]
    noun = NOUN.get(role, role)
    nn = norm(name)
    if not name or not good_name(name):
        return out
    same = [x for x in rec_els if norm(clean(x["name"])) == nn]
    ctx = e.get("ctx", {})
    unique = len(same) == 1
    if unique:
        r = rng.random()
        if r < 0.35:
            out.append((name, "name", dict(t="bare")))
        else:
            out.append((f"the {q(name)} {noun}", "name", dict(t="role")))
        b = e["box"]
        ar = (b[2] - b[0]) / max(1, b[3] - b[1])
        if e.get("icon") and role in ICON_KIND_ROLES and 0.75 <= ar <= 1.34:
            out.append((f"the {q(name)} icon" if rng.random() < 0.5 else f"{name} icon", "icon", dict(t="icon")))
    # context-qualified (also disambiguates duplicates when the context differs)
    cexp = None
    if ctx.get("menu_path") and role.endswith("menu item"):
        cexp = f"the {q(name)} item in the {ctx['menu_path'][-1]} menu"
    elif ctx.get("toolbar") and ctx["toolbar"].strip():
        cexp = f"the {q(name)} {noun} in the {ctx['toolbar'].strip()} toolbar"
    elif ctx.get("top", {}).get("role") in ("dialog", "file chooser", "alert") and ctx["top"].get("name"):
        cexp = f"the {q(name)} {noun} in the {q(ctx['top']['name'])} dialog"
    elif ctx.get("tab"):
        cexp = f"the {q(name)} {noun} on the {q(ctx['tab'])} tab"
    elif ctx.get("named_anc"):
        r_, n_ = ctx["named_anc"][0]
        if r_ in WEB_REGIONS:
            cexp = f"the {q(name)} {noun} in the {r_}" if norm(n_) == norm(r_) else f"the {q(name)} {noun} in the {q(n_)} {r_}"
        elif norm(n_) != nn:
            cexp = f"the {q(name)} {noun} in the {q(n_)} {NOUN.get(r_, r_)}"
    if cexp:
        same_ctx = [x for x in same if x is not e and x.get("ctx", {}).get("parent_i") == ctx.get("parent_i")]
        if unique or not same_ctx:
            ok = unique or len({_ctxkey(x) for x in same}) == len(same)
            if ok:
                out.append((cexp, "name", dict(t="context")))
    if not unique:  # disambiguate by screen region if regions differ
        cells = Counter(cell9(x["box"], W, H) for x in same)
        c = cell9(e["box"], W, H)
        if cells[c] == 1:
            out.append((f"the {q(name)} {noun} at the {c} of the screen", "position", dict(t="region")))
    # function: AT-SPI description / tooltip text that differs from the name
    d = good_desc(e.get("desc", ""), name)
    if d and unique:
        if not any(norm(clean(x.get("desc", ""))) == norm(d) for x in rec_els if x is not e):
            out.append((d.rstrip("."), "function", dict(t="desc")))
    return out


def _ctxkey(x):
    c = x.get("ctx", {})
    return (tuple(c.get("menu_path", [])), c.get("toolbar"), c.get("top", {}).get("name"), c.get("tab"),
            tuple(map(tuple, c.get("named_anc", [])[:1])))


def ordinal_exprs(els, W, H, tb_count=None):
    """Position-ordinal expressions for icon-like elements in toolbars: 'the third icon from the left in the top
    toolbar'. Only emitted when the toolbar location word is unique among toolbars and the toolbar is a single row/col."""
    groups = {}
    for e in els:
        c = e.get("ctx", {})
        if e.get("icon") and e["role"] in ICON_KIND_ROLES and c.get("toolbar_box") is not None:
            groups.setdefault(tuple(c["toolbar_box"]), []).append(e)
    loc_of = {}
    for tb in groups:
        x, y, w, h = tb
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        horiz = w >= h
        if horiz:
            loc = "top" if cy < 0.2 else ("bottom" if cy > 0.8 else None)
        else:
            loc = "left" if cx < 0.2 else ("right" if cx > 0.8 else None)
        loc_of[tb] = (loc, horiz)
    cnt = Counter(v[0] for v in loc_of.values() if v[0])
    out = []
    for tb, items in groups.items():
        loc, horiz = loc_of[tb]
        if not loc or cnt[loc] > 1 or len(items) < 3:
            continue
        if tb_count is None or tb_count(tb) != {e["i"] for e in items}:
            continue  # toolbar has items that are not kept icons (filtered, toggles exposed as check boxes, combos): ordinals would shift
        key = (lambda e: e["box"][0]) if horiz else (lambda e: e["box"][1])
        items = sorted(items, key=key)
        # single line check
        span = [((e["box"][1] + e["box"][3]) / 2 if horiz else (e["box"][0] + e["box"][2]) / 2) for e in items]
        if max(span) - min(span) > 8:
            continue
        side = "left" if horiz else "top"
        for k, e in enumerate(items[:10]):
            out.append((e, f"the {ORD[k]} icon from the {side} in the {loc} toolbar", "position", dict(t="ordinal", k=k)))
        if len(items) > 3:
            e = items[-1]
            out.append((e, f"the last icon in the {loc} toolbar", "position", dict(t="ordinal_last")))
    return out


class Absence:
    """Proof that an element description is absent from a screenshot's full tree (all nodes, showing or not)."""

    def __init__(self, names, texts):
        self.names = [norm(n) for n in names if n]
        self.nset = set(self.names)
        self.blob = " | ".join(self.names + [norm(t) for t in texts if t])

    def absent(self, name):
        from rapidfuzz import fuzz, process
        n = norm(name)
        if len(n) < 3 or n in self.nset or n in self.blob:
            return False
        toks = n.split()
        if toks and all(t in DOC_WORDS or t.isdigit() for t in toks):
            return False
        if self.names:
            m = process.extractOne(n, self.names, scorer=fuzz.ratio, score_cutoff=80)
            if m is not None:
                return False
        return True
