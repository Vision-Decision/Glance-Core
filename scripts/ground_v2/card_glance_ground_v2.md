---
license: cc-by-nc-4.0
language:
- en
- zh
pretty_name: glance-ground-v2
configs:
- config_name: full
  data_files:
  - split: train
    path: full/train-*.jsonl
- config_name: single
  data_files:
  - split: train
    path: single/train-*.jsonl
- config_name: diagnostic
  data_files:
  - split: test
    path: diagnostic.jsonl
- config_name: diagnostic_pos_sole
  data_files:
  - split: test
    path: diagnostic_pos_sole.jsonl
---

# glance-ground-v2

Glance v1.2 grounding training records. The data is `Vision-Decision/Glance-Data` (24,406 COCO train2014 images,
1,831,146 questions; ground truth from COCO annotations, wording by DeepSeek) converted into the Glance record format
(`dv/data/mixed.py`), plus one rule-made type that Glance-Data lacks.

| config | records | questions | use |
|---|---:|---:|---|
| `full` | 51,177 (≤48 questions each) | 1,894,336 | multi-box head: every question, `boxes` lists all targets |
| `single` | 45,027 | 1,620,520 | current single-box head: only `n_targets` ≤ 1 |
| `diagnostic` | 300 (one per image) | 23,578 | Glance-Data diagnostic split (refcoco_val images), eval only |
| `diagnostic_pos_sole` | 160 | 764 | sole-object position / superlative positives on the same images, eval only |

Record: `{"src", "img": "coco/train2014/<file>", "wh": [W, H], "max_vis": 640, "plen", "image_id", "questions": [...]}`.
Question: `{"type": "box", "text", "present", "box", "boxes", "n_targets", "qtype", "lang", "neg", "src"}`. Boxes are
pixel `[x1, y1, x2, y2]` in the original image. `box` is set only for single-target questions (backward compatible);
`boxes` is `[]` for absent, one box for single, several for multi. `qtype` = Glance-Data `kind`
(short, medium, long, casual, group, function, cat_only, cat_all, gref_multi, hneg_attribute, hneg_relation, neg_count,
neg_position, neg_near) or `pos_pos_sole`. `neg` = "hard" for every negative kind.

**Added type `pos_pos_sole` (src `gap_rule`, 63,190 train questions, en + zh).** A position word or superlative on the
only instance of a category ("the leftmost cat", "the biggest cat", "the cat on the left", "最左边的猫", "唯一的那只猫").
Glance v1 rejected 86.5% of "the leftmost X" when there is a single X. Side words are used only when the box centre is
clearly on that side and the box does not cross the middle. The instance must be at least 1% of the image, not crowd, and
not a fragment cut off by the border. The positives are sampled 1:1 against Glance-Data's `neg_position` (63,198), so
position words are not tied to either answer. Human spot check: 8/8.

Decontamination: no train image is in RefCOCO/+/g val/test*, gRefCOCO val/testA/testB, COCO val2017, the
ambiguity/attribute probe images, `Vision-Decision/glance-robust-ref`, or the diagnostic split
(`scripts/ground_v2/check_decontam.py`).

Code: `scripts/ground_v2/` in the decision-vison repo. Images are COCO 2014 train (not included). License follows
Glance-Data (CC BY-NC 4.0, because of gRefCOCO).
