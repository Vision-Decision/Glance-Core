"""(B) attribute-mismatch probe with Glance (all questions about an image in one pass)."""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from probe_lib import load, ask_with_centers, center_stats, iou
from glance import Box
S, out = sys.argv[1], sys.argv[2]
items = json.load(open(S + '/amb/b_spec.json'))
m = load(); t0 = time.time()
for it in items:
    qs = [Box(q['text']) for q in it['questions']]
    res, centers, ps, dt = ask_with_centers(m, S + '/amb/img/%012d.jpg' % it['image_id'], qs)
    for q, x, p in zip(it['questions'], res, ps):
        st = center_stats(p, centers, it['W'], it['H'], [it['gt_box']], topk=3, pred_box=x.box)
        q.update(p=x.present_prob, kappa=x.iou_prob, conf=x.confidence, box=list(x.box), iou_gt=round(iou(x.box, it['gt_box']), 3),
                 center_H=st['entropy_norm'], center_mass_gt=st['inst_mass'][0], center_mop15=st['mass_out_pred15'], center_npk=st['n_peaks'])
    it['secs'] = round(dt, 2)
    print(it['idx'], round(dt, 2), flush=True)
json.dump(dict(model='Vision-Decision/Glance-4B-preview', device=str(m.device), images='COCO val2017', wall_s=round(time.time() - t0),
               license_ids={1: 'CC BY-NC-SA 2.0', 2: 'CC BY-NC 2.0', 3: 'CC BY-NC-ND 2.0', 4: 'CC BY 2.0', 5: 'CC BY-SA 2.0', 6: 'CC BY-ND 2.0'},
               note='true_colors verified by visual inspection; present=False rows are attribute-swapped descriptions of a single-instance object',
               items=items), open(out, 'w'), indent=0)
