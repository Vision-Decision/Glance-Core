import json,collections,random,sys
S=sys.argv[1]
coco=json.load(open(S+'/media/coco/annotations/instances_val2017.json'))
cats={c['id']:c['name'] for c in coco['categories']}
ims={i['id']:i for i in coco['images']}
anns=collections.defaultdict(list)
for a in coco['annotations']: anns[a['image_id']].append(a)
CATS=['person','car','dog','cat','sheep','cow','horse','bird','zebra','elephant','giraffe','bus','motorcycle','bicycle','boat','umbrella','truck','airplane','bear','teddy bear','train']
OK_LIC={4,5,6,7,8}
multi2,multi3,single=[],[],[]
for iid,al in anns.items():
    im=ims[iid]; A=im['width']*im['height']
    by=collections.defaultdict(list)
    for a in al: by[cats[a['category_id']]].append(a)
    for c in CATS:
        L=by.get(c,[])
        if not L or any(a['iscrowd'] for a in L): continue
        fr=[a['area']/A for a in L]
        rec=dict(image_id=iid,cat=c,W=im['width'],H=im['height'],license=im['license'],flickr_url=im.get('flickr_url'),coco_url=im['coco_url'],
                 boxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in L],area=[round(f,4) for f in fr])
        if len(L)==1 and fr[0]>=0.02: single.append(rec)
        elif 2<=len(L)<=6 and min(fr)>=0.005 and sorted(fr)[-2]>=0.015:
            (multi2 if len(L)==2 else multi3).append(rec)
random.seed(1)
def pick(lst,n,capper=4,capperson=8):
    random.shuffle(lst); lst.sort(key=lambda r:r['license'] not in OK_LIC)
    out=[];cnt=collections.Counter();seen=set()
    for r in lst:
        if r['image_id'] in seen: continue
        if cnt[r['cat']]>=(capperson if r['cat']=='person' else capper): continue
        out.append(r);cnt[r['cat']]+=1;seen.add(r['image_id'])
        if len(out)>=n: break
    return out
sel=dict(multi=pick(multi2,20)+pick(multi3,20),single=pick(single,40))
print({k:len(v) for k,v in sel.items()}, collections.Counter(r['cat'] for r in sel['multi']), collections.Counter(r['cat'] for r in sel['single']))
print(collections.Counter(r['license'] for v in sel.values() for r in v))
json.dump(sel,open(S+'/amb/a_sel.json','w'))
for v in sel.values():
    for r in v: print(r['coco_url'])
