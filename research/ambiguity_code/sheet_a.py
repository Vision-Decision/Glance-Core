import json,sys
from PIL import Image, ImageDraw
S=sys.argv[1]; sel=json.load(open(S+'/amb/a_sel.json')); L=sel['multi']+sel['single']
T=256
for s in range(0,len(L),20):
    sheet=Image.new('RGB',(5*T,4*T),'white'); d=ImageDraw.Draw(sheet)
    for k,o in enumerate(L[s:s+20]):
        im=Image.open(S+'/amb/imgA/%012d.jpg'%o['image_id']).convert('RGB'); sc=T/max(im.size)
        im=im.resize((int(im.size[0]*sc),int(im.size[1]*sc))); dd=ImageDraw.Draw(im)
        for b in o['boxes']: dd.rectangle([v*sc for v in b],outline=(255,0,255),width=2)
        X,Y=(k%5)*T,(k//5)*T; sheet.paste(im,(X,Y)); d.rectangle([X,Y,X+130,Y+14],fill='black'); d.text((X+2,Y+1),f"{s+k} {o['cat']} n={len(o['boxes'])}",fill='white')
    sheet.save(S+f'/amb/sheetA{s//20}.jpg',quality=85)
