# sprint_robust: description-shift robustness (generators, recipes, eval)

Reusable generators (text only; TRAIN-split images for training data, VAL for eval):

| script | what | needs |
|---|---|---|
| `llm.py` | DeepSeek via Ollama cloud (`OLLAMA_API_KEY` from env / `.env`, never printed) | key |
| `gen_aug.py` | per training expression: zh translation + EN paraphrase (+ functional / vague rewrite for ~40%); unique-object function / generic queries and absent counterparts. Cached + resumable | `pool_natural.jsonl`, COCO instances |
| `affordance_lexicon.py` + `gen_lex.py` | no-LLM functional / generic shift from a COCO category → affordance lexicon (EN+ZH). Each class lists the categories that satisfy it, so validity is checked against every object in the image (present: no other object fits; absent: none fits). `--holdout_cats` for unseen-category tests | same |
| `build_aug_pool.py` | gen_aug caches → training records; questions tagged `aug=` (zh, para, func, vague, ufunc, ugeneric, absfunc, absgeneric, ulit) and grouped (`grp`, `anchor`) for the consistency loss | caches |
| `make_captions.py` | EN COCO captions of train-pool images + DeepSeek ZH translations for the LM loss | captions_train2014 |
| `build_groups.py` | paired eval set `data/prep/eval_rg.jsonl` (VAL only): orig / zh / para / func / vague / zh_vague per target, unique function-only / generic, absent groups | VAL annotations |

Training-side knobs (in `dv/config.py`): `w_cons` (+ `w_cons_pres/center/box`) consistency to the anchor description,
`w_lm` / `lm_file` / `lm_bs` caption LM loss. Mixing: `mix_robust.py --variants ... [--teacher teacher_iou.json]`.

Eval / analysis: `score.py` (per-shift Acc@0.5, false-absence, N-acc, overconfident error, ECE, all-correct),
`compare.py` (tables over runs, `--exclude_texts` for unseen-text slices), `base_gen.py` (base generative baseline with a
first-token presence probability; `--lora_ckpt` runs a Glance LoRA through the generative path), `check_leak.py`.

VM plumbing: `launch_vm.sh`, `vm_setup.sh`, `run_recipe.sh`, `jobs/` (recipes, serial queues), `rbg.sh`, `push_code.sh`,
`pull.sh`, `wait_for.sh`, `keepalive.sh`. One GPU job at a time per VM (concurrent jobs OOM).
