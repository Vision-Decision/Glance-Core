# Exp 5: question-only shortcut probe for the presence gate (zero training, CPU)

**Bottom line.** A text-only classifier (TF-IDF word + char n-grams → logistic regression; it sees the expression and never the image) trained on the **v1.2 training mixture** separates present from absent box questions with AUROC **0.918** on held-out mixture questions. On the **Glance-Data diagnostic set** it gets **0.93**. The cause is the count negatives: these are bare "number + noun" strings ("two clocks", "六个椅子"), and the probe assigns them P(present) = **0.014** on average, without looking at the image. The same probe trained on the v1.1-style mixture (`train_base`) is at chance on the diagnostic set (0.48) and reaches only 0.77 in distribution.

So **v1.2's diagnostic-set numbers on count, position, near and relation negatives cannot be read as grounding.** The training negatives and the evaluation negatives come from one generator, so they share a template that the text alone gives away. This is the SugarCrepe artefact. On the external sets the shortcut is small: gRefCOCO val merged 0.56 to 0.57, and the robust-ref sets 0.45 to 0.70. The one exception is robust-func, where the v1.2 probe reaches 0.70 [0.67, 0.72].

## How Glance's known presence failures relate to the text prior (B/A sets: single-instance COCO images, v1 rows)

| Failure | Glance | Text prior P(present), v1.1-style mixture | Does the text prior predict which items fail? |
|---|---|---|---|
| Count swap accepted ("the two trains") | FA 14.1% | **0.73**: the text prior says "present" | AUROC 0.70 for Glance's false accepts |
| Size swap accepted | FA 13.5% | 0.84 | 0.57 |
| Colour swap accepted | FA 0.6% | 0.43 | (too few events) |
| "the leftmost X" with one X, refused | 86.5% | **0.77**: the text prior says "present" | ρ = 0.33 |
| "the biggest X" with one X, refused | 51% | 0.52 | ρ = 0.46 |

Two conclusions follow.

- **Count false accepts line up with a text prior.** In the v1/v1.1 data, numerals mostly appear in positives (gRefCOCO multi-target, "group" expressions). So "two trains" reads as present.
- **The "leftmost" false refusals are not a text prior.** Text alone says present (0.77). The refusal is learned jointly with the image: superlatives were trained only on scenes that have same-class distractors. Fixing this needs single-instance superlative positives (rule-built, per the report's item 10), not text debiasing.

## Decisions for phase 2
- **Item 10, count minimal pairs, changes design.** Every count negative needs a textually identical positive on another image (for example "two clocks" on an image that really has two clocks). Count phrasings must also show up in positives at the same rate. The success check should be a text-only probe at about 0.5 on the new negatives, in addition to Glance's false-accept rate.
- **v1.2 / v12b agents:** report diagnostic-set gains on templated negatives only with a text-only baseline next to them, or on the external B set (`outputs/probes/ambiguity_attr/B_attribute_glance_v1.json`).

Sources:
- Training mixtures: `Vision-Decision/glance-train-data:v1.2/{train,train_base}.jsonl`
- Diagnostic set: `Vision-Decision/glance-ground-v2:diagnostic.jsonl`
- B/A sets: `outputs/probes/ambiguity_attr/`
- Code: `arch/shortcut/probe.py`; all numbers: `results.json`
