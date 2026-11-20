"""Editor GUI example + description-shift probes, Glance-4B-preview (v1) on the README GUI mock (editor.png)."""
import json, time, sys
from glance import Glance, Box, YesNo, Choice
img = "~/Developer/Glance/assets/examples/editor.png"
out = sys.argv[1]
m = Glance.from_pretrained("Vision-Decision/Glance-4B-preview")
fig_qs = [YesNo("Is a save dialog open?"), Box("the Export button"), Box("the Cancel button in the dialog"),
          Box("the Delete all button"), Box("the Print button"), Box("the Sharpen setting"),
          Choice("Task: export the photo as a PNG file. What is the next action?", ["click", "type", "scroll", "done"]),
          YesNo("Are there unsaved changes?")]
probe_qs = [Box("the Export button"), Box("导出按钮"), Box("the button that saves the picture as a PNG"),
            Box("the Save button in the dialog"), Box("对话框里的保存按钮"), Box("the button that keeps my edits"),
            Box("the Cancel button in the dialog"), Box("取消按钮"), Box("the button that closes the dialog without doing anything")]
res = {}
for name, qs in (("fig_8q", fig_qs), ("probes", probe_qs)):
    m.ask(img, qs)  # warm-up
    t = time.time(); r = m.ask(img, qs); dt = time.time() - t
    res[name] = dict(local_secs=round(dt, 2), questions=[str(q) for q in qs], results=[dict(vars(x)) for x in r])
    print(name, dt, flush=True)
res.update(model="Vision-Decision/Glance-4B-preview (v1 preview checkpoint)", device=str(m.device), image="Glance/assets/examples/editor.png (rendered from examples/gui_mock.html)")
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False, default=list)
print("done")
