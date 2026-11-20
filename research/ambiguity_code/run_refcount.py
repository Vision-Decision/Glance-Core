"""Zero-shot 'referent count' via Glance's choice primitive on gRefCOCO val merged: can the typed head tell
multi-target from single-target expressions (an expression-level ambiguity signal)?"""
import json, sys, os, random, time
from glance import Glance, Choice, YesNo
jsonl, imgdir, out, nimg = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
recs = [json.loads(l) for l in open(jsonl)]
random.Random(0).shuffle(recs)
m = Glance.from_pretrained('Vision-Decision/Glance-4B-preview')
OPTS = ['0', '1', '2', '3 or more']
rows, t0 = [], time.time()
for r in recs[:nimg]:
    rng = random.Random(r['image_id'])
    by = {0: [], 1: [], 2: []}
    for q in r['questions']: by[min(q['n_targets'], 2)].append(q)
    pick = rng.sample(by[2], min(3, len(by[2]))) + rng.sample(by[1], min(3, len(by[1]))) + rng.sample(by[0], min(2, len(by[0])))
    qs = []
    for q in pick:
        qs.append(Choice(f'How many objects in the image match the description "{q["text"]}"?', OPTS))
        qs.append(YesNo(f'Does the description "{q["text"]}" refer to more than one object in the image?'))
    if not qs: continue
    res = m.ask(os.path.join(imgdir, r['file']), qs)
    for k, q in enumerate(pick):
        c, y = res[2 * k], res[2 * k + 1]
        rows.append(dict(img=r['image_id'], sid=q.get('sent_id'), nt=q['n_targets'], probs=c.probs, p_multi_yes=y.p_yes))
    print(len(rows), round(time.time() - t0), flush=True)
json.dump(dict(model='Vision-Decision/Glance-4B-preview', set='gRefCOCO val merged, %d random images, <=3 multi / 3 single / 2 no-target per image' % nimg,
               questions=['How many objects in the image match the description "<expr>"? [0,1,2,3 or more]',
                          'Does the description "<expr>" refer to more than one object in the image?'],
               rows=rows), open(out, 'w'), separators=(',', ':'))
