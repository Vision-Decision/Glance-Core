"""(B) attribute-mismatch spec: COCO val2017 single-instance objects; true colors verified by visual check of contact sheets."""
import json, random, sys
S = sys.argv[1]
sel = json.load(open(S + '/amb/attr_sel0.json'))
DROP = {1, 40}  # 1: frisbee pink vs red ambiguous; 40: bicycle too small, color not visible
TRUE = {0:'orange white',2:'orange red',3:'blue black silver gray',4:'white',5:'blue yellow gray',6:'brown',7:'yellow orange gray',
 8:'black white',9:'black white',10:'black brown',11:'red black',12:'red white gray',13:'blue black',14:'blue yellow',15:'blue gray white',
 16:'blue white',17:'blue',18:'brown',19:'gray white brown',20:'white',21:'black',22:'black gray',23:'yellow',24:'green white',25:'white gray',
 26:'yellow',27:'red black white',28:'black',29:'green black',30:'yellow red gray',31:'yellow gray black',32:'brown',33:'white',34:'white gray',
 35:'pink white red',36:'red',37:'white orange',38:'green white gray',39:'yellow',41:'brown green gray',42:'brown orange',43:'green white red',
 44:'white',45:'black white',46:'blue pink',47:'green black',48:'red white',49:'yellow',50:'black',51:'white',52:'yellow',53:'orange black',
 54:'brown',55:'yellow',56:'white brown',57:'black yellow green blue orange red',58:'red',59:'yellow red brown',60:'orange white',
 61:'yellow white green',62:'brown',63:'white',64:'blue',65:'brown',66:'white',67:'brown white',68:'blue',69:'red',70:'green',71:'orange',
 72:'white',73:'brown yellow orange',74:'blue white',75:'red white',76:'black',77:'red',78:'red brown',79:'white'}
# a color that is NOT the object's but is prominent on another object in the scene (binding test)
DISTRACT = {51:'green',13:'white',26:'green',45:'red',68:'green',71:'yellow',4:'brown',36:'black',39:'blue',75:'black',56:'red',61:'black',16:'green',72:'black'}
ACT = {0:('lying down',['jumping','running']),4:('standing',['lying down','jumping']),6:('being ridden',['lying down','eating grass']),
 7:('perched on a branch',['flying','swimming']),9:('sitting',['running','jumping']),18:('standing',['lying down','jumping']),
 19:('standing',['flying','swimming']),21:('walking',['sleeping','lying down']),22:('lying down',['jumping','running']),
 23:('perched on a branch',['flying','swimming']),25:('eating',['jumping','lying down']),28:('sitting',['jumping','running']),
 32:('grazing',['lying down','running']),37:('sitting',['running','swimming']),42:('standing',['lying down','jumping']),
 45:('lying down',['jumping','running']),56:('lying down',['jumping','running']),57:('perched on a branch',['flying','swimming']),
 62:('jumping',['lying down','eating grass']),67:('standing',['lying down','jumping']),72:('pulling a carriage',['lying down','jumping']),
 3:('parked',['being ridden']),13:('parked',['being ridden']),29:('parked',['being ridden']),47:('parked',['being ridden']),
 50:('parked',['being ridden']),49:('being ridden',['parked']),26:('parked',['driving down a road'])}
NEAR = {'black':{'gray'},'white':{'gray','silver'},'gray':{'white','black','silver'},'brown':{'orange','red','tan'},'red':{'orange','pink','brown'},
        'yellow':{'orange'},'orange':{'yellow','red','brown'},'blue':{'purple'},'pink':{'red','purple'},'purple':{'blue','pink'},'green':set(),'silver':{'gray','white'}}
POOL = ['red','blue','green','yellow','white','black','orange','pink','purple','gray','brown']
PL = {'bus':'buses','couch':'couches','teddy bear':'teddy bears','fire hydrant':'fire hydrants','sheep':'sheep'}
pl = lambda c: PL.get(c, c + 's')
rng = random.Random(0)
items = []
for i, o in enumerate(sel):
    if i in DROP: continue
    c = o['cat']; tc = TRUE[i].split(); x, y, w, h = o['bbox']; W, H = o['W'], o['H']
    excl = set(tc) | set().union(*[NEAR.get(t, set()) for t in tc])
    pool = [p for p in POOL if p not in excl and p != DISTRACT.get(i)]
    qs = [dict(type='bare', text=f'the {c}', present=True),
          dict(type='color', text=f'the {tc[0]} {c}', present=True)]
    for col in rng.sample(pool, min(2, len(pool))):
        qs.append(dict(type='color', text=f'the {col} {c}', present=False, swap='contrast'))
    if i in DISTRACT:
        qs.append(dict(type='color', text=f'the {DISTRACT[i]} {c}', present=False, swap='scene_color'))
    qs += [dict(type='count', text=f'the two {pl(c)}', present=False), dict(type='count', text=f'the three {pl(c)}', present=False)]
    cxr = (x + w / 2) / W
    if cxr < 0.4 or cxr > 0.6:
        side, other = ('left', 'right') if cxr < 0.4 else ('right', 'left')
        qs += [dict(type='position', text=f'the {c} on the {side}', present=True), dict(type='position', text=f'the {c} on the {other}', present=False)]
    cyr = (y + h / 2) / H
    if cyr < 0.4 or cyr > 0.6:
        side, other = ('top', 'bottom') if cyr < 0.4 else ('bottom', 'top')
        qs += [dict(type='position', text=f'the {c} at the {side}', present=True), dict(type='position', text=f'the {c} at the {other}', present=False)]
    f = o['frac']
    if f >= 0.25:
        qs += [dict(type='size', text=f'the big {c}', present=True), dict(type='size', text=f'the tiny {c}', present=False)]
    elif f < 0.05:
        qs += [dict(type='size', text=f'the small {c}', present=True), dict(type='size', text=f'the huge {c}', present=False)]
    if i in ACT:
        t, sw = ACT[i]
        mk = lambda a: f'the {a} {c}' if a == 'parked' else f'the {c} {a}'
        qs.append(dict(type='action', text=mk(t), present=True))
        qs += [dict(type='action', text=mk(a), present=False) for a in sw]
    sal = 'small' if f < 0.05 else ('medium' if f < 0.25 else 'large')
    items.append(dict(idx=i, image_id=o['image_id'], coco_url='http://images.cocodataset.org/val2017/%012d.jpg' % o['image_id'],
                      license=o['license'], cat=c, true_colors=tc, area_frac=f, salience=sal,
                      gt_box=[x, y, x + w, y + h], W=W, H=H, questions=qs))
json.dump(items, open(S + '/amb/b_spec.json', 'w'), indent=0)
import collections
print(len(items), sum(len(it['questions']) for it in items), collections.Counter((q['type'], q['present']) for it in items for q in it['questions']))
