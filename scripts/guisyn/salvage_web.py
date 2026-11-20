"""Close the .tmp job files of stopped web workers: keep complete JSON lines only, rename to .jsonl."""
import glob, json, os, sys
for f in glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else '/data/guisyn/rawweb'}/jobs/*.jsonl.tmp"):
    good = []
    for l in open(f):
        try:
            json.loads(l); good.append(l)
        except Exception:
            pass
    open(f[:-4], "w").writelines(good); os.remove(f)
    print(f, len(good))
