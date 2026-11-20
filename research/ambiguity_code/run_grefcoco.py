"""Glance-4B-preview (v1) on gRefCOCO val merged (= gRefCOCO val + RefCOCO val), with patch-center statistics.
Stores compact per-expression rows (no images)."""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
from probe_lib import load, ask_with_centers, center_stats, iou
from glance import Box
jsonl, imgdir, out = sys.argv[1:4]
lim = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
m = load()
rows, t0 = [], time.time()
recs = [json.loads(l) for l in open(jsonl)][:lim]
for n, r in enumerate(recs):
    qs = [Box(q['text']) for q in r['questions']]
    res, centers, ps, dt = ask_with_centers(m, os.path.join(imgdir, r['file']), qs)
    W, H = r['W'], r['H']
    for q, x, p in zip(r['questions'], res, ps):
        nt = q['n_targets']
        gts = q['boxes'] if nt >= 2 else ([q['box']] if q['box'] else [])
        st = center_stats(p, centers, W, H, gts or None, topk=3, pred_box=x.box)
        row = dict(img=r['image_id'], sid=q.get('sent_id'), nt=nt, p=round(x.present_prob, 4), k=round(x.iou_prob, 4),
                   box=[round(v, 1) for v in x.box], H=st['entropy_norm'], ppl=st['perplexity'], top1=st['top1'], npk=st['n_peaks'], pk21=st['peak2_over_peak1'], mop=st['mass_out_pred'], mop15=st['mass_out_pred15'])
        if gts:
            row['iou'] = [round(iou(x.box, g), 3) for g in gts]
            row['mass'] = st['inst_mass']; row['mass_any'] = st['mass_in_any']
            row['area'] = [round((g[2]-g[0])*(g[3]-g[1])/(W*H), 4) for g in gts]
            if nt >= 2:
                u = [min(g[0] for g in gts), min(g[1] for g in gts), max(g[2] for g in gts), max(g[3] for g in gts)]
                row['iou_union'] = round(iou(x.box, u), 3)
        rows.append(row)
    if n % 50 == 0:
        print(n, len(rows), round(time.time() - t0), flush=True)
json.dump(dict(model='Vision-Decision/Glance-4B-preview', device=str(m.device), set='grefcoco_val_merged (gRefCOCO val + RefCOCO val)',
               fields=dict(nt='n_targets', p='calibrated P(present)', k='P(IoU>=0.5) head', H='normalized entropy of patch-center softmax',
                           ppl='perplexity of center softmax', top1='max patch prob', iou='IoU with each GT instance',
                           mass='center-softmax mass on patches inside each GT box', mass_any='mass inside union of GT boxes (pixel sets)',
                           iou_union='IoU with the union (enclosing) box of all targets', area='GT box area fraction'),
               wall_s=round(time.time() - t0), rows=rows), open(out, 'w'), separators=(',', ':'))
print('done', len(rows))
