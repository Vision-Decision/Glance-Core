"""Web pipeline: Playwright Chromium screenshots of public top sites (Tranco) with DOM-extracted element boxes.

  python web_worker.py --domains /data/guisyn/tranco/top-1m.csv --start 0 --count 500 --wid 0 --out /data/guisyn/rawweb

Politeness / safety: robots.txt checked (User-agent "*" and "guisyn-research") for every URL; <= 3 requests per site
(robots, landing page, at most one same-site link); >= 2 s between requests to the same site; sites are processed
sequentially per worker; never types into or submits forms, never clicks; pages with a visible password field or a
login/paywall URL are skipped; adult/gambling domains are skipped by keyword.
Per screenshot (viewport at scroll positions 0, ~0.9 and ~1.8 viewport heights):
  images/{site}/{id}.webp (lossless), a11y/{site}/{id}.json.gz (elements, visible texts, page HTML), and one raw
  record in jobs/{job}.jsonl (written as .tmp, renamed when the job ends).
Hard negatives (in build_shards.py) come from other viewports/pages of the same site and are verified absent from the
current viewport's visible names and texts.
"""
import argparse, asyncio, csv, gzip, json, os, random, re, sys, time, urllib.parse, urllib.robotparser, urllib.request

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/159.0.0.0 Safari/537.36 "
      "guisyn-research/0.1 (non-commercial ML research; screenshots only)")
BAD = re.compile(r"porn|sex|xxx|adult|nude|naked|cam(s|girl)|escort|hentai|fetish|xvideo|xnxx|xhamster|onlyfans|"
                 r"casino|bet(ting)?\b|gambl|poker|slot|lottery|bet365|1xbet|stake\.|torrent|pirate|warez|"
                 r"dating|hookup|milf|boob|tits|dick|fuck|camgirl|erotic|playboy|brazzers|chaturbate|stripchat|"
                 r"bongacams|livejasmin|spankbang|redtube|youporn|pornhub|tube8|motherless|rule34|e-hentai|nhentai", re.I)
INFRA = re.compile(r"(amazonaws|akamai|cloudfront|cdn|gstatic|googleapis|doubleclick|googlesyndication|apple-dns|"
                   r"msftncsi|windowsupdate|digicert|root-servers|in-addr|ntp|dns|edgekey|edgesuite|fastly|"
                   r"azureedge|trafficmanager|1e100|icloud-content|tiktokcdn|fbcdn|akadns|googleusercontent|"
                   r"app-measurement|crashlytics|scorecardresearch|adnxs|criteo|taboola|outbrain|rubiconproject)", re.I)
LOGIN = re.compile(r"login|signin|sign-in|sign_in|auth|account|paywall|subscribe|checkout|captcha|consent", re.I)
VIEWPORTS = [((1280, 800), 1), ((1440, 900), 1), ((1920, 1080), 1), ((1536, 864), 1), ((1366, 768), 1),
             ((1280, 800), 2), ((1440, 900), 2), ((1920, 1080), 2), ((1680, 1050), 1)]

EXTRACT_JS = r"""
() => {
  const W = innerWidth, H = innerHeight;
  const out = [], texts = [];
  const vis = (el) => { const s = getComputedStyle(el); return s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity || '1') > 0.15; };
  const roleOf = (el) => {
    const r = (el.getAttribute('role') || '').toLowerCase(), t = el.tagName.toLowerCase(), ty = (el.getAttribute('type') || '').toLowerCase();
    const map = {button:'push button', link:'link', tab:'page tab', menuitem:'menu item', checkbox:'check box', radio:'radio button',
      combobox:'combo box', textbox:'entry', searchbox:'entry', switch:'toggle button', img:'image', heading:'heading', option:'list item', slider:'slider'};
    if (map[r]) return map[r];
    if (t === 'a' && el.hasAttribute('href')) return 'link';
    if (t === 'button' || t === 'summary' || (t === 'input' && ['button','submit','reset','image'].includes(ty))) return 'push button';
    if (t === 'input' && ty === 'checkbox') return 'check box';
    if (t === 'input' && ty === 'radio') return 'radio button';
    if (t === 'input' && ['range'].includes(ty)) return 'slider';
    if (t === 'input' && !['hidden','file','color','date'].includes(ty)) return 'entry';
    if (t === 'textarea') return 'entry';
    if (t === 'select') return 'combo box';
    if (t === 'img' || t === 'svg') return 'image';
    if (/^h[1-6]$/.test(t)) return 'heading';
    if (t === 'label') return 'label';
    return null;
  };
  const nameOf = (el) => {
    let n = el.getAttribute('aria-label');
    if (!n && el.getAttribute('aria-labelledby')) { const l = document.getElementById(el.getAttribute('aria-labelledby').split(' ')[0]); if (l) n = l.innerText; }
    if (!n && el.tagName === 'IMG') n = el.getAttribute('alt');
    if (!n) n = (el.innerText || '').trim();
    if (!n && el.tagName === 'svg') { const t = el.querySelector('title'); if (t) n = t.textContent; }
    if (!n) n = el.getAttribute('title') || el.getAttribute('placeholder') || ((el.tagName === 'INPUT' && ['button','submit'].includes(el.type)) ? el.value : '');
    if (!n && el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) n = l.innerText; }
    return (n || '').replace(/\s+/g, ' ').trim();
  };
  const region = (el) => {
    const l = el.closest('nav,header,footer,aside,form,[role=navigation],[role=banner],[role=contentinfo],[role=search],dialog,[role=dialog]');
    if (!l) return null;
    const t = l.tagName.toLowerCase(), r = (l.getAttribute('role') || '');
    const nm = l.getAttribute('aria-label') || '';
    const kind = {nav:'navigation', header:'header', footer:'footer', aside:'sidebar', form:'form', dialog:'dialog'}[t] ||
                 {navigation:'navigation', banner:'header', contentinfo:'footer', search:'search', dialog:'dialog'}[r] || t;
    return [kind, nm];
  };
  const occl = (el, r) => {
    let ok = 0;
    const pts = [[.5,.5],[.25,.3],[.75,.3],[.25,.7],[.75,.7]];
    for (const [fx, fy] of pts) {
      const x = r.left + r.width * fx, y = r.top + r.height * fy;
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const e = document.elementFromPoint(x, y);
      if (e && (e === el || el.contains(e) || e.contains(el) && e.getBoundingClientRect().width <= r.width * 1.3)) ok++;
    }
    return ok;
  };
  const inView = (r) => r.width >= 4 && r.height >= 4 && r.left >= -2 && r.top >= -2 && r.right <= W + 2 && r.bottom <= H + 2;
  let i = 0;
  const seen = new Set();
  const all = document.querySelectorAll('a[href],button,input,select,textarea,img,svg,[role],h1,h2,h3,h4,h5,h6,label,summary,[aria-label],[title]');
  for (const el of all) {
    if (out.length > 1500) break;
    const role = roleOf(el); if (!role) continue;
    if (el.tagName.toLowerCase() === 'svg' && el.closest('a,button')) continue;
    const r = el.getBoundingClientRect();
    const name = nameOf(el).slice(0, 200);
    const rec = {i: i++, role, name, box: [r.left, r.top, r.right, r.bottom], vis: false};
    if (inView(r) && vis(el) && occl(el, r) >= 4) rec.vis = true;
    else if (!(r.width >= 4 && r.height >= 4 && vis(el))) continue;  // not rendered at all
    rec.icon = (r.width <= 48 && r.height <= 48 && r.width / r.height > 0.5 && r.width / r.height < 2 && !(el.innerText || '').trim());
    const rg = region(el); if (rg) rec.region = rg;
    const d = el.getAttribute('title'); if (d && d !== name) rec.desc = d.slice(0, 160);
    seen.add(el); out.push(rec);
  }
  // short visible text leaves (static text); also collect every visible text for absence proofs
  const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n; const leafSeen = new Set();
  while ((n = tw.nextNode())) {
    const t = n.textContent.replace(/\s+/g, ' ').trim(); if (!t) continue;
    const el = n.parentElement; if (!el || leafSeen.has(el)) continue;
    const rg = document.createRange(); rg.selectNodeContents(n); const r = rg.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.top > H || r.right < 0 || r.left > W || !vis(el)) continue;
    texts.push(t.slice(0, 200));
    if (seen.has(el) || el.closest('a,button,label,h1,h2,h3,h4,h5,h6,[role=button],[role=link]')) continue;
    if (t.length > 60 || t.length < 2 || !inView(r)) continue;
    leafSeen.add(el);
    out.push({i: i++, role: 'static', name: t, box: [r.left, r.top, r.right, r.bottom], vis: occl(el, r) >= 4, region: region(el)});
  }
  const pw = !!Array.from(document.querySelectorAll('input[type=password]')).find(e => { const r = e.getBoundingClientRect(); return r.width > 0 && vis(e); });
  return {W, H, els: out, texts, password: pw, title: document.title, sh: document.documentElement.scrollHeight,
          links: Array.from(document.querySelectorAll('a[href]')).map(a => a.href).filter(h => h.startsWith(location.origin)).slice(0, 200)};
}
"""


class Robots:
    def __init__(self):
        self.cache = {}

    def ok(self, url):
        p = urllib.parse.urlsplit(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                req = urllib.request.Request(base + "/robots.txt", headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=8) as r:
                    if r.status >= 400:
                        raise IOError
                    rp.parse(r.read(500_000).decode("utf8", "ignore").splitlines())
                self.cache[base] = rp
            except urllib.error.HTTPError as e:
                self.cache[base] = "allow" if e.code in (404, 410) else None  # 401/403/5xx -> treat as disallow
            except Exception:
                self.cache[base] = None
        rp = self.cache[base]
        if rp == "allow":
            return True
        if rp is None:
            return False
        return rp.can_fetch("guisyn-research", url) and rp.can_fetch("*", url)


async def shoot(page, site, url, vp, dpr, k, a, recf, rng, stats):
    """One page: screenshots at up to 3 scroll offsets."""
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
    except Exception:
        stats["goto_fail"] += 1; return []
    if resp is None or resp.status >= 400:
        stats["http_err"] += 1; return []
    if LOGIN.search(urllib.parse.urlsplit(page.url).path + urllib.parse.urlsplit(page.url).netloc.split(".")[0]):
        stats["login_url"] += 1; return []
    await page.wait_for_timeout(2500)
    links = []
    n_saved = 0
    for si, frac in enumerate([0, 0.9, 1.8][: rng.choice([1, 2, 3, 3])]):
        try:
            if frac:
                await page.evaluate(f"window.scrollTo(0, {int(vp[1] * frac)})")
                await page.wait_for_timeout(900)
            d = await page.evaluate(EXTRACT_JS)
        except Exception:
            stats["eval_fail"] += 1; break
        if d["password"]:
            stats["password_page"] += 1; return []
        if frac and d["sh"] < vp[1] * (frac + 0.5):
            break
        links = d["links"]
        vis = [e for e in d["els"] if e["vis"]]
        named = [e for e in vis if e["name"]]
        if len(named) < 5:
            stats["too_few"] += 1; continue
        sid = f"{site.replace('.', '_')}-{k:02d}{si}-{vp[0]}x{vp[1]}d{dpr}"
        ip = f"images/{site}/{sid}.webp"
        os.makedirs(f"{a.out}/images/{site}", exist_ok=True); os.makedirs(f"{a.out}/a11y/{site}", exist_ok=True)
        png = await page.screenshot(type="png")
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(png)).convert("RGB")
        W, H = im.size
        im.save(f"{a.out}/{ip}", "WEBP", lossless=True, quality=50, method=2)
        html = await page.content()
        apath = f"a11y/{site}/{sid}.json.gz"
        with gzip.open(f"{a.out}/{apath}", "wt") as f:
            json.dump(dict(url=page.url, title=d["title"], viewport=vp, dpr=dpr, scroll=frac, els=d["els"], texts=d["texts"],
                           html=html[:3_000_000]), f)
        s = dpr
        els = []
        for e in vis:
            x1, y1, x2, y2 = [max(0, v) * s for v in e["box"]]
            x2, y2 = min(x2, W), min(y2, H)
            if x2 - x1 < 6 or y2 - y1 < 6 or (x2 - x1) * (y2 - y1) > 0.35 * W * H:
                continue
            ctx = dict(top=dict(role="document web", name=d["title"][:80]))
            if e.get("region"):
                kind, nm = e["region"]
                ctx["named_anc"] = [(kind if kind in ("navigation", "header", "footer", "sidebar", "form", "dialog", "search") else "region", nm or kind)]
            el = dict(i=e["i"], role=e["role"], name=e["name"], box=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                      states=["sensitive"], icon=bool(e.get("icon")), depth=0, ctx=ctx)
            if e.get("desc"):
                el["desc"] = e["desc"]
            if not e["name"]:
                if e.get("icon") or e["role"] == "image":
                    el["uncaptioned"] = True
                else:
                    continue
            els.append(el)
        # region phrases: "in the footer" etc. are only kept when the region kind is unique on screen (see exprs)
        rec = dict(id=sid, app=site, source="web", url=page.url, W=W, H=H, dpr=dpr, scale=dpr, step=si, image=ip, a11y=apath,
                   elements=els, t=time.time(), viewport=vp, scroll=frac,
                   all_names=sorted({(e["role"], e["name"]) for e in vis if e["name"]}))
        recf.write(json.dumps(rec) + "\n"); recf.flush()
        n_saved += 1
        stats["shots"] += 1
        stats["elements"] += len(els)
    return links if n_saved else []


async def run(a):
    from playwright.async_api import async_playwright
    rng = random.Random(a.wid * 7919 + int(time.time()))
    rows = list(csv.reader(open(a.domains)))[a.start:a.start + a.count]
    sites = [d for _, d in rows if not BAD.search(d) and not INFRA.search(d)]
    os.makedirs(f"{a.out}/jobs", exist_ok=True); os.makedirs(f"{a.out}/logs", exist_ok=True)
    job = f"web-w{a.wid}-{int(time.time())}"
    robots = Robots()
    from collections import Counter
    stats = Counter()
    log = open(f"{a.out}/logs/{job}.log", "w")
    t_end = time.time() + a.max_s
    async with async_playwright() as p:
        br = await p.chromium.launch(args=["--disable-gpu", "--mute-audio"])
        recf, nrec, part = None, 0, 0
        for k, site in enumerate(sites):
            if time.time() > t_end or os.path.exists("/data/guisyn/STOP_WEB"):
                break
            if recf is None:
                recf = open(f"{a.out}/jobs/{job}-{part:03d}.jsonl.tmp", "w")
            url = f"https://{site}/"
            if not robots.ok(url):
                stats["robots_disallow"] += 1; print("robots", site, file=log, flush=True); continue
            vp, dpr = rng.choice(VIEWPORTS)
            ctx = await br.new_context(viewport=dict(width=vp[0], height=vp[1]), device_scale_factor=dpr, user_agent=UA,
                                       java_script_enabled=True, locale="en-US", accept_downloads=False,
                                       permissions=[], ignore_https_errors=False)
            ctx.set_default_timeout(20000)
            page = await ctx.new_page()
            try:
                links = await asyncio.wait_for(shoot(page, site, url, vp, dpr, 0, a, recf, rng, stats), 60)
                stats["sites"] += 1
                links = [l for l in links if not LOGIN.search(l) and not re.search(r"\.(pdf|zip|exe|dmg|jpg|png|mp4)$", l, re.I)]
                if links and rng.random() < 0.6:
                    l = rng.choice(links)
                    if robots.ok(l):
                        await asyncio.sleep(2.0)
                        await asyncio.wait_for(shoot(page, site, l, vp, dpr, 1, a, recf, rng, stats), 60)
            except Exception as ex:
                stats["error"] += 1
                print("err", site, repr(ex)[:200], file=log, flush=True)
            finally:
                try:
                    await ctx.close()
                except Exception:
                    pass
            if stats["shots"] - nrec >= 300:  # rotate job shard
                recf.close(); os.rename(recf.name, recf.name[:-4]); recf = None; part += 1; nrec = stats["shots"]
            if k % 20 == 0:
                print(time.strftime("%T"), k, dict(stats), file=log, flush=True)
            await asyncio.sleep(0.5)
        if recf is not None:
            recf.close(); os.rename(recf.name, recf.name[:-4])
        await br.close()
    print("done", dict(stats), file=log, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="/data/guisyn/tranco/top-1m.csv")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--wid", type=int, default=0); ap.add_argument("--out", default="/data/guisyn/rawweb")
    ap.add_argument("--max_s", type=int, default=3600 * 6)
    asyncio.run(run(ap.parse_args()))
