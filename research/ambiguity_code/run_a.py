"""(A) ambiguity probe: bare category vs disambiguated references on COCO val2017 images with 1 or >=2 instances."""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from probe_lib import load, ask_with_centers, center_stats, iou
from glance import Box, Choice, YesNo
S, out = sys.argv[1], sys.argv[2]
PL = {'person': 'people', 'bus': 'buses', 'sheep': 'sheep', 'teddy bear': 'teddy bears'}
plural = lambda c: PL.get(c, c + 's')
DROP_MULTI, DROP_SINGLE = {34}, {50, 51, 57}   # indices in a_sel (visual check: hidden/collage/painted)
sel = json.load(open(S + '/amb/a_sel.json'))
items = [('multi', r) for i, r in enumerate(sel['multi']) if i not in DROP_MULTI] + \
        [('single', r) for i, r in enumerate(sel['single']) if i + 40 not in DROP_SINGLE]
m = load()
COUNT_OPTS = ['0', '1', '2', '3', '4', '5', '6 or more']
res_all, t0 = [], time.time()
for kind, r in items:
    c, B, W, H = r['cat'], r['boxes'], r['W'], r['H']
    n = len(B)
    union = [min(b[0] for b in B), min(b[1] for b in B), max(b[2] for b in B), max(b[3] for b in B)]
    cx = [(b[0] + b[2]) / 2 for b in B]; ar = [(b[2] - b[0]) * (b[3] - b[1]) for b in B]
    order = sorted(range(n), key=lambda i: cx[i])
    qs = [(Box(f'the {c}'), 'bare_the', None), (Box(f'a {c}'), 'bare_a', None), (Box(f'the {plural(c)}'), 'plural', 'union')]
    if n == 1 or cx[order[1]] - cx[order[0]] >= 0.08 * W:
        qs.append((Box(f'the leftmost {c}'), 'leftmost', order[0]))
    if n == 1 or cx[order[-1]] - cx[order[-2]] >= 0.08 * W:
        qs.append((Box(f'the rightmost {c}'), 'rightmost', order[-1]))
    sa = sorted(range(n), key=lambda i: -ar[i])
    if n == 1 or ar[sa[0]] >= 1.5 * ar[sa[1]]:
        qs.append((Box(f'the biggest {c}'), 'biggest', sa[0]))
    qs.append((Choice(f'How many {plural(c)} are there?', COUNT_OPTS), 'count', None))
    qs.append((YesNo(f'Is there more than one {c}?'), 'more_than_one', None))
    img = S + '/amb/imgA/%012d.jpg' % r['image_id']
    res, centers, ps, dt = ask_with_centers(m, img, [q for q, _, _ in qs])
    rows = []
    for (q, tag, gt), x, p in zip(qs, res, ps):
        row = dict(tag=tag, text=getattr(q, 'text', None) or q.question)
        if tag == 'count':
            row.update(probs=x.probs, answer=x.answer)
        elif tag == 'more_than_one':
            row.update(p_yes=x.p_yes)
        else:
            st = center_stats(p, centers, W, H, B, topk=5, pred_box=x.box)
            row.update(gt=gt, p=x.present_prob, kappa=x.iou_prob, conf=x.confidence, box=list(x.box),
                       iou_inst=[round(iou(x.box, b), 3) for b in B], iou_union=round(iou(x.box, union), 3), center=st)
        rows.append(row)
    res_all.append(dict(kind=kind, image_id=r['image_id'], coco_url=r['coco_url'], flickr_url=r['flickr_url'], license=r['license'],
                        cat=c, n_inst=n, W=W, H=H, gt_boxes=[[round(v, 1) for v in b] for b in B], area_frac=r['area'],
                        secs=round(dt, 2), questions=rows))
    print(len(res_all), kind, c, n, round(dt, 2), flush=True)
lic = {1: 'CC BY-NC-SA 2.0', 2: 'CC BY-NC 2.0', 3: 'CC BY-NC-ND 2.0', 4: 'CC BY 2.0', 5: 'CC BY-SA 2.0', 6: 'CC BY-ND 2.0'}
json.dump(dict(model='Vision-Decision/Glance-4B-preview', device=str(m.device), images='COCO val2017 (not in Glance training; see report)',
               license_ids=lic, note='center: patch-center softmax stats; inst_mass = softmax mass on patches whose centers lie in each GT box',
               wall_s=round(time.time() - t0), items=res_all), open(out, 'w'), indent=0)
