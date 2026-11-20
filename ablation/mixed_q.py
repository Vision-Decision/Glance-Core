"""Mixed typed workload shared by ablation/latency_mixed.py (Glance v1, our engine) and ablation/latency_vllm.py
--mixed (generative baseline): image i = i-th record of eval_refcoco_val.jsonl; question j of Q cycles through
the types box, yes/no, choice, score, built from the record's referring expressions (cycled). The typed prompt bodies
are those of dv/data/mixed.ChatTok.typed_block, so both systems read the same text."""
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TYPES = ["box", "noul", "choice", "score"]
POS = ["left", "center", "right"]
SIZE = ["tiny", "small", "medium", "large", "very large"]


def mixed_questions(exprs, Q):
    out = []
    for j in range(Q):
        e = exprs[(j // 4) % len(exprs)] if Q > 4 else exprs[j % len(exprs)]
        t = TYPES[j % 4]
        if t == "box":
            out.append(dict(type="box", text=e))
        elif t == "noul":
            out.append(dict(type="noul", instr=f'Is there "{e}" in the image?', target=0.5))
        elif t == "choice":
            out.append(dict(type="choice", instr=f'Where is "{e}" in the image?', options=POS, target=[1 / 3] * 3))
        else:
            out.append(dict(type="score", instr=f'How large is "{e}" in the image?', options=SIZE, target=[.2] * 5))
    return out


def typed_body(q):
    """= dv/data/mixed.ChatTok.typed_block body text."""
    if q["type"] == "noul":
        return q["instr"].strip() + "\nAnswer yes or no."
    head = "Options:" if q["type"] == "choice" else "Scale (lowest to highest):"
    tail = "Answer with the letter of the best option." if q["type"] == "choice" else "Answer with the letter of the level."
    return q["instr"].strip() + f"\n\n{head}\n" + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(q["options"])) + tail
