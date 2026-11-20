"""Post-hoc IoU head on the frozen Gen-v1 (recipe of ablation/ar_iou_head.py, unchanged: LayerNorm'd [h_q | h_ans | h_end]
+ standardized scalars + Fourier(box centre) -> Linear 512 -> GELU -> Linear 1, BCE on 1[IoU >= .5], AdamW 3e-4, 40
epochs, early stopping on a 10% image-held-out slice). Trained only on H = arhead_rcg (RefCOCOg val+test images outside
gRefCOCO val/testA/testB; never seen in v1/Gen-v1 training). The in-sample variant (T) of the controlled study is not
run: Gen-v1's training images are the v1 mixture, and the held-out head is the one the comparison uses.

python -m gen_v1.iou_head --feat_dir DIR/feats --out DIR/head_scores.npz
"""
import ablation.ar_iou_head as H

H.VARIANTS = {"heldout_full": ("arhead_rcg", True), "heldout_scalar": ("arhead_rcg", False)}

if __name__ == "__main__":
    H.main()
