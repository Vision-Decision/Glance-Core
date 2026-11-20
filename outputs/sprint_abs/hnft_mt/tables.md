# sprint_abs tables (hnft_mt)

## GREC official protocol (gRefCOCO; Pr@(F1=1, GIoU>=0.5), N-acc, T-acc)

| split | variant | Pr@F1=1 | 95% CI | N-acc | T-acc | Pr single | Pr multi | Pr (plain IoU) |
|---|---|---|---|---|---|---|---|---|
| grec_val | single_raw@0.7(official) | 71.6 | 71.6 [71.1, 72.1] | 89.4 | 96.6 | 92.1 | 0.0 | 71.8 |
| grec_val | single_raw@0.5 | 70.4 | 70.4 [69.9, 70.9] | 85.2 | 98.2 | 92.9 | 0.0 | 70.6 |
| grec_val | single_platt@0.7 | 71.7 | 71.7 [71.2, 72.2] | 90.0 | 96.2 | 91.9 | 0.0 | 71.9 |
| grec_val | multi_platt@0.7(official) | 71.7 | 71.7 [71.2, 72.2] | 90.0 | 96.2 | 91.9 | 0.0 | 71.9 |
| grec_val | multi_raw@0.7(official) | 71.6 | 71.6 [71.1, 72.1] | 89.4 | 96.6 | 92.1 | 0.0 | 71.8 |
| grec_val | multi_platt@tuned | 70.6 | 70.6 [70.1, 71.1] | 85.0 | 98.2 | 92.5 | 2.0 | 70.9 |
| grec_val | single_platt@tuned | 70.4 | 70.4 [69.8, 70.8] | 85.0 | 98.2 | 92.9 | 0.0 | 70.6 |
| grec_val | paired diff multi-single (official) | 0.1 [0.1, 0.2] | | | | | | |
| grec_testA | single_raw@0.7(official) | 47.2 | 47.2 [46.4, 47.9] | 82.9 | 93.6 | 90.7 | 0.0 | 47.3 |
| grec_testA | single_raw@0.5 | 46.3 | 46.3 [45.6, 47.0] | 78.1 | 95.7 | 91.4 | 0.0 | 46.4 |
| grec_testA | single_platt@0.7 | 47.3 | 47.3 [46.6, 48.0] | 83.5 | 93.2 | 90.6 | 0.0 | 47.4 |
| grec_testA | multi_platt@0.7(official) | 47.3 | 47.3 [46.6, 48.0] | 83.5 | 93.2 | 90.6 | 0.0 | 47.4 |
| grec_testA | multi_raw@0.7(official) | 47.2 | 47.2 [46.4, 47.9] | 82.9 | 93.6 | 90.7 | 0.0 | 47.3 |
| grec_testA | multi_platt@tuned | 46.8 | 46.8 [46.0, 47.6] | 78.0 | 95.7 | 91.2 | 1.4 | 47.1 |
| grec_testA | single_platt@tuned | 46.3 | 46.3 [45.5, 47.0] | 78.0 | 95.7 | 91.5 | 0.0 | 46.4 |
| grec_testA | paired diff multi-single (official) | 0.1 [0.0, 0.2] | | | | | | |
| grec_testB | single_raw@0.7(official) | 51.2 | 51.2 [50.1, 52.3] | 79.6 | 89.1 | 79.9 | 0.0 | 51.6 |
| grec_testB | single_raw@0.5 | 50.1 | 50.1 [49.1, 51.2] | 75.0 | 91.0 | 80.6 | 0.0 | 50.5 |
| grec_testB | single_platt@0.7 | 51.4 | 51.4 [50.3, 52.5] | 80.3 | 88.6 | 79.6 | 0.0 | 51.7 |
| grec_testB | multi_platt@0.7(official) | 51.4 | 51.4 [50.3, 52.5] | 80.3 | 88.6 | 79.6 | 0.0 | 51.7 |
| grec_testB | multi_raw@0.7(official) | 51.2 | 51.2 [50.1, 52.3] | 79.6 | 89.1 | 79.9 | 0.0 | 51.6 |
| grec_testB | multi_platt@tuned | 50.3 | 50.3 [49.2, 51.4] | 74.7 | 91.0 | 80.0 | 1.2 | 50.7 |
| grec_testB | single_platt@tuned | 50.0 | 50.0 [49.0, 51.1] | 74.7 | 91.0 | 80.6 | 0.0 | 50.4 |
| grec_testB | paired diff multi-single (official) | 0.1 [0.1, 0.2] | | | | | | |
| grefcalib | single_raw@0.7(official) | 63.2 | - | 96.5 | 97.7 | 91.4 | 0.0 | 63.7 |
| grefcalib | single_raw@0.5 | 63.4 | - | 95.8 | 98.6 | 91.8 | 0.0 | 63.9 |
| grefcalib | single_platt@0.7 | 63.1 | - | 96.7 | 97.3 | 91.2 | 0.0 | 63.6 |
| grefcalib | multi_platt@0.7(official) | 63.1 | - | 96.7 | 97.3 | 91.2 | 0.0 | 63.6 |
| grefcalib | multi_raw@0.7(official) | 63.2 | - | 96.5 | 97.7 | 91.4 | 0.0 | 63.7 |
| grefcalib | multi_platt@tuned | 63.9 | - | 95.8 | 98.7 | 91.5 | 2.3 | 64.6 |
| grefcalib | single_platt@tuned | 63.4 | - | 95.8 | 98.7 | 91.8 | 0.0 | 63.9 |

Tuned on grefcalib: {"pr_f1_grefcalib": 0.6390478370336461, "t_presence": 0.44999999999999996, "t_extra": 0.2, "single_t_presence": 0.44999999999999996, "single_pr_f1_grefcalib": 0.6337834744792858}; extra-box model: {"n": 1537, "pos": 0.12036434612882238}

## Presence ĝ calibration (fit on held-out training-distribution splits)

| set | subset | method | AUROC | N-acc@.5 | recall@.5 | ECE15 | ECE15-mass | ECE10 | Brier | NLL | +EM prior: ECE15 / N-acc / π̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grec_val | all | raw | 98.0 | 85.2 | 98.2 | 3.7 | 3.6 | 3.7 | 5.01 | 0.187 | - |
| grec_val | all | temp | 98.0 | 85.2 | 98.2 | 3.7 | 3.6 | 3.6 | 5.00 | 0.186 | 3.0 / 86.5 / 0.673 |
| grec_val | all | platt | 98.0 | 86.1 | 97.9 | 3.3 | 3.2 | 3.3 | 4.89 | 0.182 | 2.6 / 87.4 / 0.668 |
| grec_val | all | beta | 98.0 | 86.3 | 97.9 | 3.2 | 3.2 | 3.2 | 4.87 | 0.183 | 2.5 / 87.5 / 0.668 |
| grec_val | all | iso | 98.0 | 86.2 | 97.9 | 3.2 | 3.2 | 3.2 | 4.91 | 0.184 | 2.6 / 87.5 / 0.669 |
| grec_val | all | multi | 97.9 | 85.4 | 98.0 | 3.2 | 3.2 | 3.2 | 4.91 | 0.174 | 2.1 / 87.0 / 0.666 |
| grec_val | gref_only | raw | 96.5 | 85.2 | 96.4 | 7.9 | 7.9 | 7.9 | 8.21 | 0.305 | - |
| grec_val | gref_only | temp | 96.5 | 85.2 | 96.4 | 7.9 | 7.9 | 7.9 | 8.19 | 0.304 | 2.6 / 91.6 / 0.388 |
| grec_val | gref_only | platt | 96.5 | 86.1 | 96.0 | 7.2 | 7.2 | 7.2 | 7.93 | 0.295 | 2.7 / 92.0 / 0.381 |
| grec_val | gref_only | beta | 96.5 | 86.3 | 95.8 | 7.1 | 7.1 | 7.1 | 7.89 | 0.295 | 2.7 / 92.1 / 0.381 |
| grec_val | gref_only | iso | 96.5 | 86.2 | 95.9 | 7.2 | 7.2 | 7.2 | 7.97 | 0.299 | 2.4 / 91.6 / 0.381 |
| grec_val | gref_only | multi | 96.5 | 85.4 | 96.2 | 7.6 | 7.6 | 7.6 | 7.98 | 0.278 | 2.1 / 92.9 / 0.365 |
| grec_testA | all | raw | 94.1 | 78.1 | 95.7 | 4.3 | 4.0 | 4.3 | 6.76 | 0.300 | - |
| grec_testA | all | temp | 94.1 | 78.1 | 95.7 | 4.3 | 3.9 | 4.2 | 6.76 | 0.297 | 4.5 / 76.6 / 0.783 |
| grec_testA | all | platt | 94.1 | 79.2 | 95.4 | 4.3 | 4.2 | 4.3 | 6.79 | 0.301 | 4.5 / 77.7 / 0.778 |
| grec_testA | all | beta | 94.1 | 79.4 | 95.3 | 4.3 | 4.2 | 4.3 | 6.80 | 0.299 | 4.4 / 77.8 / 0.778 |
| grec_testA | all | iso | 94.1 | 79.3 | 95.3 | 4.4 | 4.1 | 4.4 | 6.83 | 0.292 | 4.5 / 77.3 / 0.778 |
| grec_testA | all | multi | 94.1 | 78.1 | 95.6 | 3.3 | 3.2 | 3.2 | 6.68 | 0.269 | 3.6 / 76.5 / 0.779 |
| grec_testA | gref_only | raw | 93.3 | 78.1 | 95.0 | 5.3 | 4.9 | 5.3 | 8.38 | 0.335 | - |
| grec_testA | gref_only | temp | 93.3 | 78.1 | 95.0 | 5.2 | 4.9 | 5.2 | 8.37 | 0.332 | 5.0 / 79.4 / 0.683 |
| grec_testA | gref_only | platt | 93.3 | 79.2 | 94.6 | 5.2 | 5.1 | 5.1 | 8.39 | 0.333 | 5.0 / 80.5 / 0.676 |
| grec_testA | gref_only | beta | 93.3 | 79.4 | 94.6 | 5.2 | 5.1 | 5.1 | 8.38 | 0.334 | 5.0 / 80.6 / 0.676 |
| grec_testA | gref_only | iso | 93.3 | 79.3 | 94.6 | 5.3 | 5.1 | 5.2 | 8.43 | 0.333 | 5.1 / 80.6 / 0.677 |
| grec_testA | gref_only | multi | 93.3 | 78.1 | 95.0 | 4.0 | 3.7 | 4.0 | 8.26 | 0.304 | 3.7 / 79.9 / 0.673 |
| grec_testB | all | raw | 89.0 | 75.0 | 91.0 | 9.2 | 9.0 | 9.1 | 11.56 | 0.621 | - |
| grec_testB | all | temp | 89.0 | 75.0 | 91.0 | 9.1 | 9.0 | 9.0 | 11.55 | 0.614 | 9.0 / 75.7 / 0.705 |
| grec_testB | all | platt | 89.0 | 76.0 | 90.7 | 8.9 | 8.8 | 8.9 | 11.55 | 0.625 | 8.8 / 76.6 / 0.699 |
| grec_testB | all | beta | 89.0 | 76.2 | 90.6 | 8.9 | 8.8 | 8.9 | 11.54 | 0.613 | 8.8 / 76.8 / 0.699 |
| grec_testB | all | iso | 89.0 | 76.1 | 90.6 | 9.1 | 8.9 | 9.1 | 11.59 | 0.585 | 8.9 / 76.7 / 0.699 |
| grec_testB | all | multi | 88.9 | 75.1 | 90.9 | 8.0 | 7.9 | 7.9 | 11.38 | 0.535 | 7.7 / 76.2 / 0.695 |
| grec_testB | gref_only | raw | 90.2 | 75.0 | 92.9 | 9.1 | 8.9 | 9.1 | 12.14 | 0.499 | - |
| grec_testB | gref_only | temp | 90.2 | 75.0 | 92.9 | 9.0 | 8.9 | 9.0 | 12.11 | 0.495 | 8.0 / 78.6 / 0.584 |
| grec_testB | gref_only | platt | 90.2 | 76.0 | 92.4 | 8.9 | 8.5 | 8.9 | 12.05 | 0.492 | 8.1 / 79.3 / 0.575 |
| grec_testB | gref_only | beta | 90.2 | 76.2 | 92.3 | 8.8 | 8.4 | 8.8 | 12.03 | 0.494 | 8.1 / 79.6 / 0.575 |
| grec_testB | gref_only | iso | 90.2 | 76.1 | 92.4 | 9.1 | 8.5 | 9.1 | 12.12 | 0.497 | 8.0 / 78.9 / 0.576 |
| grec_testB | gref_only | multi | 90.1 | 75.1 | 92.8 | 7.8 | 7.5 | 7.8 | 11.84 | 0.441 | 6.2 / 79.8 / 0.564 |
| calib_natural | all | raw | 99.5 | 94.1 | 98.9 | 0.4 | 0.5 | 0.4 | 1.92 | 0.072 | - |
| calib_natural | all | temp | 99.5 | 94.1 | 98.9 | 0.4 | 0.5 | 0.4 | 1.92 | 0.072 | 0.5 / 94.1 / 0.726 |
| calib_natural | all | platt | 99.5 | 94.4 | 98.8 | 0.4 | 0.4 | 0.2 | 1.91 | 0.071 | 0.4 / 94.4 / 0.723 |
| calib_natural | all | beta | 99.5 | 94.5 | 98.7 | 0.3 | 0.4 | 0.2 | 1.91 | 0.071 | 0.3 / 94.5 / 0.723 |
| calib_natural | all | iso | 99.5 | 94.4 | 98.7 | 0.3 | 0.2 | 0.2 | 1.90 | 0.071 | 0.3 / 94.4 / 0.723 |
| calib_natural | all | multi | 99.5 | 94.4 | 98.8 | 1.0 | 1.0 | 1.0 | 1.93 | 0.074 | 1.0 / 94.4 / 0.721 |
| grefcalib | all | raw | 99.4 | 95.8 | 98.6 | 1.7 | 1.8 | 1.7 | 1.26 | 0.049 | - |
| grefcalib | all | temp | 99.4 | 95.8 | 98.6 | 1.8 | 1.8 | 1.8 | 1.26 | 0.049 | 0.3 / 92.9 / 0.895 |
| grefcalib | all | platt | 99.4 | 95.8 | 98.5 | 2.0 | 2.0 | 2.0 | 1.35 | 0.052 | 0.5 / 92.9 / 0.893 |
| grefcalib | all | beta | 99.4 | 95.8 | 98.5 | 2.0 | 2.1 | 2.0 | 1.36 | 0.052 | 0.4 / 92.9 / 0.894 |
| grefcalib | all | iso | 99.4 | 95.8 | 98.5 | 2.0 | 2.0 | 2.0 | 1.35 | 0.052 | 0.4 / 92.9 / 0.894 |
| grefcalib | all | multi | 99.4 | 96.0 | 98.6 | 2.5 | 2.5 | 2.5 | 1.31 | 0.055 | 0.9 / 91.6 / 0.895 |
| guisyn_test | natural_fit | raw | 99.3 | 96.3 | 97.5 | 0.9 | 0.9 | 0.9 | 2.30 | 0.088 | - |
| guisyn_test | natural_fit | temp | 99.3 | 96.3 | 97.5 | 0.9 | 0.9 | 0.9 | 2.30 | 0.088 | 0.7 / 95.9 / 0.740 |
| guisyn_test | natural_fit | platt | 99.3 | 96.6 | 97.3 | 1.2 | 1.2 | 1.2 | 2.34 | 0.090 | 1.1 / 96.4 / 0.737 |
| guisyn_test | natural_fit | beta | 99.3 | 96.7 | 97.2 | 1.2 | 1.2 | 1.2 | 2.34 | 0.089 | 1.0 / 96.5 / 0.737 |
| guisyn_test | natural_fit | iso | 99.3 | 96.6 | 97.2 | 1.1 | 1.1 | 1.1 | 2.34 | 0.088 | 1.0 / 96.3 / 0.738 |
| guisyn_test | natural_fit | multi | 99.3 | 91.7 | 98.3 | 2.2 | 2.1 | 2.2 | 2.69 | 0.097 | 3.0 / 89.5 / 0.776 |
| guisyn_test | gui_fit | raw | 99.3 | 96.3 | 97.5 | 0.9 | 0.9 | 0.9 | 2.30 | 0.088 | - |
| guisyn_test | gui_fit | temp | 99.3 | 96.3 | 97.5 | 0.9 | 0.9 | 0.9 | 2.30 | 0.087 | 1.2 / 96.6 / 0.736 |
| guisyn_test | gui_fit | platt | 99.3 | 95.0 | 98.0 | 0.6 | 0.6 | 0.5 | 2.28 | 0.085 | 0.5 / 95.2 / 0.747 |
| guisyn_test | gui_fit | beta | 99.3 | 94.8 | 98.0 | 0.7 | 0.6 | 0.6 | 2.28 | 0.086 | 0.6 / 95.0 / 0.747 |
| guisyn_test | gui_fit | iso | 99.3 | 94.4 | 98.1 | 0.6 | 0.6 | 0.5 | 2.28 | 0.085 | 0.5 / 94.5 / 0.747 |
| guisyn_test | gui_fit | multi | 99.3 | 94.9 | 97.9 | 0.8 | 0.7 | 0.7 | 2.35 | 0.088 | 0.9 / 95.1 / 0.744 |
| venus | natural_fit | raw | 73.2 | 54.9 | 75.3 | 19.3 | 19.4 | 19.2 | 20.68 | 0.714 | - |
| venus | natural_fit | temp | 73.2 | 54.9 | 75.3 | 19.3 | 19.4 | 19.2 | 20.63 | 0.709 | 22.0 / 61.5 / 0.647 |
| venus | natural_fit | platt | 73.2 | 57.3 | 73.7 | 20.7 | 20.6 | 20.6 | 21.46 | 0.745 | 24.4 / 66.2 / 0.620 |
| venus | natural_fit | beta | 73.2 | 57.6 | 73.4 | 21.0 | 20.8 | 20.7 | 21.54 | 0.739 | 24.8 / 67.1 / 0.617 |
| venus | natural_fit | iso | 73.2 | 57.4 | 73.5 | 20.3 | 20.5 | 20.2 | 21.71 | 0.732 | 24.2 / 65.5 / 0.623 |
| venus | natural_fit | multi | 74.9 | 41.4 | 85.1 | 11.2 | 11.2 | 11.0 | 15.16 | 0.480 | 9.2 / 32.8 / 0.808 |
| venus | gui_fit | raw | 73.2 | 54.9 | 75.3 | 19.3 | 19.4 | 19.2 | 20.68 | 0.714 | - |
| venus | gui_fit | temp | 73.2 | 54.9 | 75.3 | 19.2 | 19.1 | 19.0 | 20.39 | 0.684 | 26.2 / 68.9 / 0.599 |
| venus | gui_fit | platt | 73.2 | 47.1 | 79.4 | 15.8 | 15.8 | 15.8 | 18.45 | 0.621 | 18.2 / 52.3 / 0.692 |
| venus | gui_fit | beta | 73.2 | 46.4 | 80.0 | 15.5 | 15.5 | 15.5 | 18.35 | 0.624 | 17.6 / 51.1 / 0.698 |
| venus | gui_fit | iso | 73.2 | 45.3 | 80.7 | 15.7 | 15.6 | 15.7 | 18.60 | 0.620 | 16.8 / 47.4 / 0.710 |
| venus | gui_fit | multi | 70.3 | 46.5 | 76.1 | 18.0 | 18.0 | 18.0 | 20.00 | 0.634 | 23.6 / 59.5 / 0.623 |

## κ̂ calibration (natural: event IoU>=0.5; GUI: click in box)

| set | fit | method | AUROC | AURC | ECE15 | ECE15-mass | Brier | mean κ̂ | rate |
|---|---|---|---|---|---|---|---|---|---|
| refcoco_val | natural_fit | raw | 89.5 | 1.03 | 0.5 | 0.7 | 4.20 | 93.9 | 94.0 |
| refcoco_val | natural_fit | temp | 89.5 | 1.03 | 0.6 | 0.7 | 4.20 | 93.9 | 94.0 |
| refcoco_val | natural_fit | platt | 89.5 | 1.03 | 0.5 | 0.8 | 4.22 | 93.7 | 94.0 |
| refcoco_val | natural_fit | beta | 89.5 | 1.03 | 0.7 | 0.8 | 4.22 | 93.6 | 94.0 |
| refcoco_val | natural_fit | iso | 89.5 | 1.03 | 0.8 | 0.8 | 4.21 | 93.6 | 94.0 |
| refcoco_val | natural_fit | multi | 89.6 | 1.03 | 0.9 | 0.8 | 4.19 | 93.2 | 94.0 |
| refcocog_test | natural_fit | raw | 89.1 | 1.61 | 0.6 | 0.6 | 5.63 | 91.4 | 91.4 |
| refcocog_test | natural_fit | temp | 89.1 | 1.61 | 0.6 | 0.6 | 5.63 | 91.4 | 91.4 |
| refcocog_test | natural_fit | platt | 89.1 | 1.61 | 0.9 | 0.8 | 5.65 | 91.0 | 91.4 |
| refcocog_test | natural_fit | beta | 89.1 | 1.61 | 0.8 | 0.9 | 5.65 | 90.9 | 91.4 |
| refcocog_test | natural_fit | iso | 89.1 | 1.61 | 1.0 | 1.1 | 5.66 | 90.8 | 91.4 |
| refcocog_test | natural_fit | multi | 89.1 | 1.62 | 0.8 | 0.7 | 5.64 | 90.8 | 91.4 |
| grec_val | natural_fit | raw | 89.4 | 1.02 | 0.6 | 0.7 | 4.18 | 93.9 | 94.1 |
| grec_val | natural_fit | temp | 89.4 | 1.02 | 0.6 | 0.7 | 4.18 | 93.9 | 94.1 |
| grec_val | natural_fit | platt | 89.4 | 1.02 | 0.7 | 0.8 | 4.20 | 93.7 | 94.1 |
| grec_val | natural_fit | beta | 89.4 | 1.02 | 0.8 | 0.9 | 4.20 | 93.6 | 94.1 |
| grec_val | natural_fit | iso | 89.4 | 1.02 | 0.9 | 1.0 | 4.19 | 93.6 | 94.1 |
| grec_val | natural_fit | multi | 89.5 | 1.03 | 1.0 | 0.9 | 4.17 | 93.2 | 94.1 |
| grec_testA | natural_fit | raw | 93.3 | 0.83 | 1.4 | 1.8 | 4.00 | 94.4 | 93.0 |
| grec_testA | natural_fit | temp | 93.3 | 0.83 | 1.4 | 1.8 | 4.00 | 94.3 | 93.0 |
| grec_testA | natural_fit | platt | 93.3 | 0.83 | 1.2 | 1.5 | 3.96 | 94.1 | 93.0 |
| grec_testA | natural_fit | beta | 93.3 | 0.83 | 1.3 | 1.5 | 3.96 | 94.2 | 93.0 |
| grec_testA | natural_fit | iso | 93.3 | 0.83 | 1.5 | 1.3 | 3.97 | 94.2 | 93.0 |
| grec_testA | natural_fit | multi | 93.4 | 0.80 | 1.2 | 1.6 | 3.99 | 93.8 | 93.0 |
| grec_testB | natural_fit | raw | 93.4 | 2.65 | 2.3 | 2.6 | 6.87 | 86.1 | 83.9 |
| grec_testB | natural_fit | temp | 93.4 | 2.65 | 2.2 | 2.5 | 6.87 | 86.0 | 83.9 |
| grec_testB | natural_fit | platt | 93.4 | 2.65 | 1.7 | 2.0 | 6.77 | 85.4 | 83.9 |
| grec_testB | natural_fit | beta | 93.4 | 2.65 | 2.6 | 2.4 | 6.90 | 85.6 | 83.9 |
| grec_testB | natural_fit | iso | 93.4 | 2.65 | 2.5 | 2.4 | 6.84 | 85.3 | 83.9 |
| grec_testB | natural_fit | multi | 93.5 | 2.61 | 2.2 | 2.2 | 6.79 | 85.2 | 83.9 |
| calib_natural | natural_fit | raw | 91.6 | 1.20 | 0.9 | 0.8 | 5.20 | 92.0 | 91.6 |
| calib_natural | natural_fit | temp | 91.6 | 1.20 | 1.0 | 0.8 | 5.19 | 91.9 | 91.6 |
| calib_natural | natural_fit | platt | 91.6 | 1.20 | 1.0 | 0.8 | 5.19 | 91.6 | 91.6 |
| calib_natural | natural_fit | beta | 91.6 | 1.20 | 1.1 | 0.6 | 5.19 | 91.6 | 91.6 |
| calib_natural | natural_fit | iso | 91.6 | 1.20 | 0.5 | 0.3 | 5.16 | 91.6 | 91.6 |
| calib_natural | natural_fit | multi | 91.8 | 1.17 | 1.1 | 0.9 | 5.14 | 91.3 | 91.6 |
| guisyn_test | natural_fit | raw | 87.7 | 3.10 | 19.8 | 19.8 | 13.11 | 66.8 | 86.7 |
| guisyn_test | natural_fit | temp | 87.7 | 3.10 | 19.9 | 19.9 | 13.11 | 66.8 | 86.7 |
| guisyn_test | natural_fit | platt | 87.7 | 3.10 | 21.3 | 21.3 | 13.97 | 65.3 | 86.7 |
| guisyn_test | natural_fit | beta | 87.7 | 3.10 | 21.0 | 21.0 | 13.60 | 65.7 | 86.7 |
| guisyn_test | natural_fit | iso | 87.6 | 3.16 | 21.7 | 21.7 | 13.95 | 64.9 | 86.7 |
| guisyn_test | natural_fit | multi | 88.1 | 3.03 | 63.0 | 63.0 | 51.05 | 23.6 | 86.7 |
| guisyn_test | gui_fit | raw | 87.7 | 3.10 | 19.8 | 19.8 | 13.11 | 66.8 | 86.7 |
| guisyn_test | gui_fit | temp | 87.7 | 3.10 | 17.8 | 17.8 | 12.96 | 68.9 | 86.7 |
| guisyn_test | gui_fit | platt | 87.7 | 3.10 | 1.4 | 1.3 | 7.76 | 86.9 | 86.7 |
| guisyn_test | gui_fit | beta | 87.7 | 3.10 | 1.3 | 1.3 | 7.75 | 86.9 | 86.7 |
| guisyn_test | gui_fit | iso | 87.7 | 3.10 | 1.3 | 1.3 | 7.76 | 86.8 | 86.7 |
| guisyn_test | gui_fit | multi | 88.7 | 2.91 | 1.3 | 1.0 | 7.51 | 86.8 | 86.7 |
| ss2 | natural_fit | raw | 79.6 | 2.85 | 22.9 | 22.9 | 15.91 | 68.0 | 91.0 |
| ss2 | natural_fit | temp | 79.6 | 2.85 | 23.0 | 23.0 | 15.90 | 68.0 | 91.0 |
| ss2 | natural_fit | platt | 79.6 | 2.85 | 24.4 | 24.4 | 16.97 | 66.5 | 91.0 |
| ss2 | natural_fit | beta | 79.6 | 2.85 | 24.0 | 24.0 | 15.90 | 67.0 | 91.0 |
| ss2 | natural_fit | iso | 79.5 | 2.86 | 24.7 | 24.7 | 16.71 | 66.3 | 91.0 |
| ss2 | natural_fit | multi | 82.9 | 2.44 | 64.7 | 64.7 | 52.61 | 26.3 | 91.0 |
| ss2 | gui_fit | raw | 79.6 | 2.85 | 22.9 | 22.9 | 15.91 | 68.0 | 91.0 |
| ss2 | gui_fit | temp | 79.6 | 2.85 | 21.3 | 21.3 | 16.30 | 69.7 | 91.0 |
| ss2 | gui_fit | platt | 79.6 | 2.85 | 4.5 | 4.5 | 8.24 | 88.6 | 91.0 |
| ss2 | gui_fit | beta | 79.6 | 2.85 | 4.3 | 4.5 | 8.23 | 88.6 | 91.0 |
| ss2 | gui_fit | iso | 79.6 | 2.85 | 4.7 | 4.7 | 8.28 | 88.4 | 91.0 |
| ss2 | gui_fit | multi | 83.4 | 2.39 | 3.8 | 4.3 | 7.45 | 87.6 | 91.0 |
| venus | natural_fit | raw | 83.4 | 25.61 | 7.1 | 7.1 | 17.37 | 43.7 | 50.1 |
| venus | natural_fit | temp | 83.4 | 25.61 | 7.0 | 7.0 | 17.36 | 43.7 | 50.1 |
| venus | natural_fit | platt | 83.4 | 25.61 | 8.8 | 8.7 | 17.73 | 41.9 | 50.1 |
| venus | natural_fit | beta | 83.4 | 25.61 | 6.8 | 7.0 | 17.25 | 45.2 | 50.1 |
| venus | natural_fit | iso | 82.9 | 26.09 | 6.7 | 7.2 | 17.62 | 43.7 | 50.1 |
| venus | natural_fit | multi | 83.2 | 25.69 | 38.3 | 38.3 | 35.01 | 11.8 | 50.1 |
| venus | gui_fit | raw | 83.4 | 25.61 | 7.1 | 7.1 | 17.37 | 43.7 | 50.1 |
| venus | gui_fit | temp | 83.4 | 25.61 | 9.4 | 9.2 | 17.91 | 43.3 | 50.1 |
| venus | gui_fit | platt | 83.4 | 25.61 | 17.8 | 17.8 | 20.32 | 67.9 | 50.1 |
| venus | gui_fit | beta | 83.4 | 25.61 | 17.9 | 17.9 | 20.30 | 68.0 | 50.1 |
| venus | gui_fit | iso | 83.4 | 25.61 | 17.7 | 17.7 | 20.24 | 67.8 | 50.1 |
| venus | gui_fit | multi | 83.9 | 25.44 | 14.7 | 14.7 | 19.21 | 64.8 | 50.1 |

## Box confidence ĝ·κ̂ vs (present ∧ IoU>=0.5)  [the v0→v1 regression metric]

| set | method | ECE15 | ECE15-mass | Brier | π̂ (EM) |
|---|---|---|---|---|---|
| refcoco_val | raw | 1.71 | 1.83 | 4.74 | - |
| refcoco_val | evalpy_T | 1.74 | 1.83 | 4.74 | - |
| refcoco_val | g:platt|k:platt | 2.11 | 2.18 | 4.83 | - |
| refcoco_val | g:platt+em|k:platt | 0.56 | 0.82 | 4.22 | 0.999 |
| refcoco_val | g:platt|k:iso | 2.15 | 2.38 | 4.83 | - |
| refcoco_val | g:platt+em|k:iso | 0.86 | 0.87 | 4.21 | 0.999 |
| refcoco_val | g:platt|k:multi | 2.52 | 2.52 | 4.82 | - |
| refcoco_val | g:platt+em|k:multi | 0.95 | 0.84 | 4.20 | 0.999 |
| refcoco_val | g:iso|k:platt | 2.10 | 2.20 | 4.83 | - |
| refcoco_val | g:iso+em|k:platt | 0.55 | 0.82 | 4.22 | 0.999 |
| refcoco_val | g:iso|k:iso | 2.17 | 2.35 | 4.82 | - |
| refcoco_val | g:iso+em|k:iso | 0.85 | 0.86 | 4.21 | 0.999 |
| refcoco_val | g:iso|k:multi | 2.50 | 2.50 | 4.81 | - |
| refcoco_val | g:iso+em|k:multi | 0.93 | 0.83 | 4.19 | 0.999 |
| refcoco_val | kappa_only_raw | 0.55 | 0.74 | 4.20 | - |
| refcocog_test | raw | 2.88 | 2.88 | 6.41 | - |
| refcocog_test | evalpy_T | 2.92 | 2.92 | 6.41 | - |
| refcocog_test | g:platt|k:platt | 3.51 | 3.52 | 6.56 | - |
| refcocog_test | g:platt+em|k:platt | 0.90 | 0.76 | 5.65 | 0.999 |
| refcocog_test | g:platt|k:iso | 3.69 | 3.86 | 6.59 | - |
| refcocog_test | g:platt+em|k:iso | 1.02 | 1.13 | 5.67 | 0.999 |
| refcocog_test | g:platt|k:multi | 3.74 | 3.74 | 6.57 | - |
| refcocog_test | g:platt+em|k:multi | 0.81 | 0.80 | 5.65 | 0.999 |
| refcocog_test | g:iso|k:platt | 3.51 | 3.51 | 6.56 | - |
| refcocog_test | g:iso+em|k:platt | 0.91 | 0.79 | 5.65 | 0.999 |
| refcocog_test | g:iso|k:iso | 3.68 | 3.81 | 6.59 | - |
| refcocog_test | g:iso+em|k:iso | 0.98 | 1.12 | 5.66 | 0.999 |
| refcocog_test | g:iso|k:multi | 3.73 | 3.73 | 6.57 | - |
| refcocog_test | g:iso+em|k:multi | 0.80 | 0.79 | 5.64 | 0.999 |
| refcocog_test | kappa_only_raw | 0.62 | 0.56 | 5.63 | - |
| grec_val | raw | 5.11 | 5.12 | 6.42 | - |
| grec_val | evalpy_T | 5.11 | 5.12 | 6.41 | - |
| grec_val | g:platt|k:platt | 4.54 | 4.55 | 6.23 | - |
| grec_val | g:platt+em|k:platt | 3.20 | 3.21 | 5.74 | 0.591 |
| grec_val | g:platt|k:iso | 4.50 | 4.51 | 6.19 | - |
| grec_val | g:platt+em|k:iso | 3.16 | 3.17 | 5.72 | 0.591 |
| grec_val | g:platt|k:multi | 4.30 | 4.31 | 6.18 | - |
| grec_val | g:platt+em|k:multi | 2.96 | 2.97 | 5.70 | 0.591 |
| grec_val | g:iso|k:platt | 4.57 | 4.58 | 6.24 | - |
| grec_val | g:iso+em|k:platt | 3.24 | 3.25 | 5.76 | 0.591 |
| grec_val | g:iso|k:iso | 4.52 | 4.53 | 6.21 | - |
| grec_val | g:iso+em|k:iso | 3.20 | 3.21 | 5.74 | 0.591 |
| grec_val | g:iso|k:multi | 4.32 | 4.33 | 6.20 | - |
| grec_val | g:iso+em|k:multi | 3.00 | 3.01 | 5.73 | 0.591 |
| grec_val | kappa_only_raw | 28.86 | 28.86 | 23.90 | - |
| grec_testA | raw | 7.88 | 8.00 | 8.41 | - |
| grec_testA | evalpy_T | 7.87 | 7.98 | 8.40 | - |
| grec_testA | g:platt|k:platt | 7.38 | 7.51 | 8.17 | - |
| grec_testA | g:platt+em|k:platt | 6.42 | 6.56 | 7.70 | 0.627 |
| grec_testA | g:platt|k:iso | 7.40 | 7.53 | 8.15 | - |
| grec_testA | g:platt+em|k:iso | 6.45 | 6.58 | 7.69 | 0.627 |
| grec_testA | g:platt|k:multi | 7.18 | 7.29 | 8.14 | - |
| grec_testA | g:platt+em|k:multi | 6.23 | 6.35 | 7.68 | 0.627 |
| grec_testA | g:iso|k:platt | 7.22 | 7.44 | 8.19 | - |
| grec_testA | g:iso+em|k:platt | 6.39 | 6.52 | 7.73 | 0.627 |
| grec_testA | g:iso|k:iso | 7.23 | 7.46 | 8.17 | - |
| grec_testA | g:iso+em|k:iso | 6.37 | 6.55 | 7.72 | 0.627 |
| grec_testA | g:iso|k:multi | 7.02 | 7.23 | 8.16 | - |
| grec_testA | g:iso+em|k:multi | 6.19 | 6.31 | 7.71 | 0.627 |
| grec_testA | kappa_only_raw | 29.97 | 29.96 | 25.05 | - |
| grec_testB | raw | 8.97 | 8.97 | 11.10 | - |
| grec_testB | evalpy_T | 8.93 | 8.97 | 11.09 | - |
| grec_testB | g:platt|k:platt | 8.36 | 8.38 | 10.87 | - |
| grec_testB | g:platt+em|k:platt | 6.94 | 6.92 | 10.17 | 0.574 |
| grec_testB | g:platt|k:iso | 8.21 | 8.23 | 10.81 | - |
| grec_testB | g:platt+em|k:iso | 6.84 | 6.78 | 10.13 | 0.574 |
| grec_testB | g:platt|k:multi | 8.18 | 8.20 | 10.82 | - |
| grec_testB | g:platt+em|k:multi | 6.78 | 6.74 | 10.13 | 0.574 |
| grec_testB | g:iso|k:platt | 8.25 | 8.28 | 10.89 | - |
| grec_testB | g:iso+em|k:platt | 6.94 | 6.89 | 10.20 | 0.574 |
| grec_testB | g:iso|k:iso | 8.12 | 8.15 | 10.83 | - |
| grec_testB | g:iso+em|k:iso | 6.80 | 6.77 | 10.16 | 0.574 |
| grec_testB | g:iso|k:multi | 8.07 | 8.12 | 10.85 | - |
| grec_testB | g:iso+em|k:multi | 6.77 | 6.71 | 10.16 | 0.574 |
| grec_testB | kappa_only_raw | 31.47 | 31.47 | 27.08 | - |

Bootstrap (image clusters): 
- refcoco_val: raw ECE 1.71 [1.39, 2.28]; fixed 0.56 [0.53, 1.16]; paired diff -1.16 [-1.33, -0.57]; κ̂ AUROC 89.5 [87.6, 91.3]
- refcocog_test: raw ECE 2.88 [2.37, 3.42]; fixed 0.90 [0.78, 1.57]; paired diff -1.98 [-2.30, -1.16]; κ̂ AUROC 89.1 [87.9, 90.3]

## Presence bootstrap CIs (image clusters, raw ĝ unless noted)

- grec_val: AUROC 98.0 [97.8, 98.1]; N-acc 85.2 [84.6, 86.0]; ECE raw 3.69 [3.43, 3.99] → Platt 3.27 [2.99, 3.57] (paired -0.42 [-0.47, -0.36]); subset ECE raw 7.86 [7.42, 8.33] → Platt 7.16 [6.73, 7.63]
- grec_testA: AUROC 94.1 [93.6, 94.6]; N-acc 78.1 [76.8, 79.4]; ECE raw 4.32 [3.98, 4.89] → Platt 4.28 [3.96, 4.93] (paired -0.04 [-0.15, 0.15]); subset ECE raw 4.32 [3.98, 4.89] → Platt 4.28 [3.96, 4.93]
- grec_testB: AUROC 89.0 [87.7, 90.4]; N-acc 75.0 [73.9, 76.2]; ECE raw 9.17 [8.03, 10.27] → Platt 8.94 [7.83, 10.11] (paired -0.22 [-0.27, -0.06]); subset ECE raw 9.17 [8.03, 10.27] → Platt 8.94 [7.83, 10.11]
- venus: AUROC 73.2 [71.3, 74.9]; N-acc 54.9 [51.1, 58.2]; ECE raw 19.32 [18.03, 20.60] → Platt 15.83 [14.76, 17.14] (paired -3.49 [-3.81, -2.85]); subset ECE raw 19.32 [18.03, 20.60] → Platt 15.83 [14.76, 17.14]

## Risk-controlled accept/abstain (δ=0.1; 50/50 image splits ×reps; coverage = fraction answered)


**refcoco_val** (n=10834, absent 0.0%, correct-if-answered-all 94.0%); default ĝ≥.5: coverage 99.0, selective risk 5.8, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.4 | 0.03 | 0.02 |
| selective[g] | 0.02 | 36.8 | 1.23 | 0.12 |
| selective[g] | 0.05 | 86.7 | 4.14 | 0.09 |
| selective[g] | 0.1 | 100.0 | 5.94 | 0.0 |
| joint[g] | 0.01 | 51.4 | 0.81 | 0.2 |
| joint[g] | 0.02 | 68.7 | 1.77 | 0.31 |
| joint[g] | 0.05 | 91.9 | 4.54 | 0.25 |
| joint[g] | 0.1 | 100.0 | 5.95 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 38.8 | 1.28 | False |
| selective_transfer[g] | 0.05 | 75.7 | 3.21 | False |
| selective_transfer[g] | 0.1 | 97.5 | 5.63 | False |
| selective[g*k] | 0.01 | 24.7 | 0.38 | 0.07 |
| selective[g*k] | 0.02 | 78.1 | 1.54 | 0.11 |
| selective[g*k] | 0.05 | 95.8 | 4.19 | 0.07 |
| selective[g*k] | 0.1 | 100.0 | 5.93 | 0.0 |
| joint[g*k] | 0.01 | 70.5 | 0.79 | 0.18 |
| joint[g*k] | 0.02 | 85.2 | 1.81 | 0.26 |
| joint[g*k] | 0.05 | 97.1 | 4.50 | 0.25 |
| joint[g*k] | 0.1 | 100.0 | 5.95 | 0.0 |
| selective_transfer[g*k] | 0.01 | 53.5 | 0.64 | False |
| selective_transfer[g*k] | 0.02 | 71.5 | 1.20 | False |
| selective_transfer[g*k] | 0.05 | 90.2 | 2.94 | False |
| selective_transfer[g*k] | 0.1 | 98.2 | 5.11 | False |

AURC ĝ 2.10 vs ĝκ̂ 1.14; paired diff -0.96 [-1.17, -0.73]

**refcocog_test** (n=9602, absent 0.0%, correct-if-answered-all 91.4%); default ĝ≥.5: coverage 98.0, selective risk 8.1, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 3.0 | 0.23 | 0.03 |
| selective[g] | 0.05 | 62.2 | 4.06 | 0.05 |
| selective[g] | 0.1 | 99.9 | 8.58 | 0.0 |
| joint[g] | 0.01 | 38.3 | 0.81 | 0.19 |
| joint[g] | 0.02 | 52.7 | 1.76 | 0.24 |
| joint[g] | 0.05 | 78.1 | 4.60 | 0.21 |
| joint[g] | 0.1 | 100.0 | 8.60 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 18.8 | 1.16 | False |
| selective_transfer[g] | 0.05 | 56.5 | 3.47 | False |
| selective_transfer[g] | 0.1 | 95.3 | 7.76 | False |
| selective[g*k] | 0.01 | 5.4 | 0.14 | 0.01 |
| selective[g*k] | 0.02 | 63.7 | 1.45 | 0.05 |
| selective[g*k] | 0.05 | 88.2 | 4.26 | 0.1 |
| selective[g*k] | 0.1 | 99.9 | 8.58 | 0.0 |
| joint[g*k] | 0.01 | 61.6 | 0.79 | 0.16 |
| joint[g*k] | 0.02 | 75.2 | 1.74 | 0.2 |
| joint[g*k] | 0.05 | 91.2 | 4.58 | 0.2 |
| joint[g*k] | 0.1 | 100.0 | 8.60 | 0.0 |
| selective_transfer[g*k] | 0.01 | 40.4 | 0.72 | False |
| selective_transfer[g*k] | 0.02 | 58.9 | 1.08 | False |
| selective_transfer[g*k] | 0.05 | 84.6 | 3.51 | False |
| selective_transfer[g*k] | 0.1 | 96.8 | 7.09 | False |

AURC ĝ 3.52 vs ĝκ̂ 1.77; paired diff -1.75 [-2.05, -1.46]

**grec_val** (n=25063, absent 35.5%, correct-if-answered-all 57.1%); default ĝ≥.5: coverage 68.5, selective risk 17.9, false-accept on absent 14.8

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.6 | 0.20 | 0.03 |
| selective[g] | 0.1 | 45.4 | 8.92 | 0.09 |
| joint[g] | 0.01 | 18.3 | 0.88 | 0.25 |
| joint[g] | 0.02 | 30.4 | 1.81 | 0.17 |
| joint[g] | 0.05 | 48.7 | 4.70 | 0.2 |
| joint[g] | 0.1 | 64.3 | 9.54 | 0.28 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 18.7 | 4.77 | True |
| selective_transfer[g] | 0.05 | 41.3 | 8.13 | True |
| selective_transfer[g] | 0.1 | 64.9 | 15.26 | True |
| selective[g*k] | 0.01 | 0.1 | 0.02 | 0.01 |
| selective[g*k] | 0.02 | 4.0 | 0.49 | 0.03 |
| selective[g*k] | 0.05 | 40.9 | 4.25 | 0.05 |
| selective[g*k] | 0.1 | 55.0 | 9.14 | 0.08 |
| joint[g*k] | 0.01 | 31.9 | 0.89 | 0.19 |
| joint[g*k] | 0.02 | 41.3 | 1.81 | 0.21 |
| joint[g*k] | 0.05 | 54.1 | 4.70 | 0.24 |
| joint[g*k] | 0.1 | 64.3 | 9.46 | 0.09 |
| selective_transfer[g*k] | 0.01 | 25.3 | 2.22 | True |
| selective_transfer[g*k] | 0.02 | 35.6 | 3.33 | True |
| selective_transfer[g*k] | 0.05 | 51.8 | 7.63 | True |
| selective_transfer[g*k] | 0.1 | 67.2 | 16.86 | True |
| absent_conformal[g] | 0.01 | present accepted 57.4 | false-accept absent 1.02 | |
| absent_conformal[g] | 0.02 | present accepted 73.5 | false-accept absent 2.05 | |
| absent_conformal[g] | 0.05 | present accepted 90.3 | false-accept absent 5.08 | |
| absent_conformal[g] | 0.1 | present accepted 96.1 | false-accept absent 10.02 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 86.7 | false-accept absent 3.85 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 92.5 | false-accept absent 6.09 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 97.7 | false-accept absent 13.05 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 99.2 | false-accept absent 22.17 | |

AURC ĝ 15.54 vs ĝκ̂ 13.45; paired diff -2.09 [-2.48, -1.77]

**grec_testA** (n=19200, absent 23.2%, correct-if-answered-all 61.6%); default ĝ≥.5: coverage 78.6, selective risk 23.3, false-accept on absent 21.9

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 5.8 | 0.89 | 0.3 |
| joint[g] | 0.02 | 15.8 | 1.84 | 0.27 |
| joint[g] | 0.05 | 35.3 | 4.66 | 0.19 |
| joint[g] | 0.1 | 57.1 | 9.64 | 0.31 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 20.7 | 11.50 | True |
| selective_transfer[g] | 0.05 | 44.7 | 14.43 | True |
| selective_transfer[g] | 0.1 | 74.6 | 21.34 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 29.2 | 3.91 | 0.0 |
| selective[g*k] | 0.1 | 49.1 | 9.01 | 0.12 |
| joint[g*k] | 0.01 | 25.0 | 0.86 | 0.17 |
| joint[g*k] | 0.02 | 35.4 | 1.82 | 0.17 |
| joint[g*k] | 0.05 | 49.8 | 4.59 | 0.1 |
| joint[g*k] | 0.1 | 63.1 | 9.43 | 0.25 |
| selective_transfer[g*k] | 0.01 | 24.6 | 3.43 | True |
| selective_transfer[g*k] | 0.02 | 33.8 | 4.86 | True |
| selective_transfer[g*k] | 0.05 | 52.7 | 10.34 | True |
| selective_transfer[g*k] | 0.1 | 75.8 | 21.50 | True |
| absent_conformal[g] | 0.01 | present accepted 22.4 | false-accept absent 0.96 | |
| absent_conformal[g] | 0.02 | present accepted 35.2 | false-accept absent 1.93 | |
| absent_conformal[g] | 0.05 | present accepted 66.0 | false-accept absent 4.82 | |
| absent_conformal[g] | 0.1 | present accepted 85.1 | false-accept absent 10.08 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 81.5 | false-accept absent 8.54 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 88.5 | false-accept absent 12.01 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 95.0 | false-accept absent 19.92 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 97.4 | false-accept absent 27.90 | |

AURC ĝ 19.06 vs ĝκ̂ 13.28; paired diff -5.79 [-6.59, -4.99]

**grec_testB** (n=16063, absent 29.1%, correct-if-answered-all 52.1%); default ĝ≥.5: coverage 71.8, selective risk 30.2, false-accept on absent 25.0

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 6.7 | 0.86 | 0.2 |
| joint[g] | 0.02 | 12.9 | 1.78 | 0.21 |
| joint[g] | 0.05 | 26.8 | 4.62 | 0.17 |
| joint[g] | 0.1 | 44.8 | 9.62 | 0.3 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 12.2 | 13.39 | True |
| selective_transfer[g] | 0.05 | 35.3 | 19.32 | True |
| selective_transfer[g] | 0.1 | 67.8 | 28.10 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 22.5 | 8.17 | 0.04 |
| joint[g*k] | 0.01 | 13.3 | 0.87 | 0.16 |
| joint[g*k] | 0.02 | 22.2 | 1.82 | 0.22 |
| joint[g*k] | 0.05 | 36.9 | 4.68 | 0.23 |
| joint[g*k] | 0.1 | 50.5 | 9.42 | 0.16 |
| selective_transfer[g*k] | 0.01 | 14.2 | 6.72 | True |
| selective_transfer[g*k] | 0.02 | 24.1 | 8.57 | True |
| selective_transfer[g*k] | 0.05 | 43.7 | 15.48 | True |
| selective_transfer[g*k] | 0.1 | 67.4 | 27.68 | True |
| absent_conformal[g] | 0.01 | present accepted 11.4 | false-accept absent 1.01 | |
| absent_conformal[g] | 0.02 | present accepted 22.6 | false-accept absent 2.04 | |
| absent_conformal[g] | 0.05 | present accepted 46.5 | false-accept absent 5.02 | |
| absent_conformal[g] | 0.1 | present accepted 72.4 | false-accept absent 9.94 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 75.9 | false-accept absent 11.04 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 83.7 | false-accept absent 14.55 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 90.4 | false-accept absent 23.22 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 93.0 | false-accept absent 31.74 | |

AURC ĝ 25.36 vs ĝκ̂ 21.28; paired diff -4.08 [-4.85, -3.46]

**ss2** (n=1272, absent 0.0%, correct-if-answered-all 91.0%); default ĝ≥.5: coverage 94.4, selective risk 7.4, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 11.0 | 0.74 | 0.01 |
| selective[g] | 0.1 | 93.5 | 7.21 | 0.01 |
| joint[g] | 0.01 | 23.1 | 0.45 | 0.1 |
| joint[g] | 0.02 | 43.8 | 1.18 | 0.1 |
| joint[g] | 0.05 | 78.4 | 3.88 | 0.15 |
| joint[g] | 0.1 | 98.3 | 8.34 | 0.09 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.1 | 71.1 | 4.20 | False |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 1.0 | 0.04 | 0.01 |
| selective[g*k] | 0.05 | 61.4 | 2.57 | 0.01 |
| selective[g*k] | 0.1 | 95.5 | 7.28 | 0.01 |
| joint[g*k] | 0.01 | 37.7 | 0.47 | 0.09 |
| joint[g*k] | 0.02 | 57.4 | 1.24 | 0.13 |
| joint[g*k] | 0.05 | 83.3 | 3.92 | 0.15 |
| joint[g*k] | 0.1 | 98.8 | 8.35 | 0.09 |
| selective_transfer[g*k] | 0.01 | 14.5 | 1.09 | True |
| selective_transfer[g*k] | 0.02 | 41.7 | 1.13 | False |
| selective_transfer[g*k] | 0.05 | 72.7 | 3.46 | False |
| selective_transfer[g*k] | 0.1 | 94.7 | 6.98 | False |

AURC ĝ 3.50 vs ĝκ̂ 2.41; paired diff -1.09 [-1.89, -0.41]

**venus** (n=6165, absent 14.6%, correct-if-answered-all 42.8%); default ĝ≥.5: coverage 70.9, selective risk 46.8, false-accept on absent 45.1

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 3.4 | 0.79 | 0.22 |
| joint[g] | 0.02 | 7.7 | 1.67 | 0.16 |
| joint[g] | 0.05 | 19.4 | 4.43 | 0.18 |
| joint[g] | 0.1 | 32.7 | 9.02 | 0.08 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.1 | 36.4 | 29.25 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 0.2 | 0.74 | 0.04 |
| joint[g*k] | 0.01 | 7.4 | 0.75 | 0.1 |
| joint[g*k] | 0.02 | 13.2 | 1.76 | 0.23 |
| joint[g*k] | 0.05 | 24.2 | 4.42 | 0.17 |
| joint[g*k] | 0.1 | 35.5 | 9.17 | 0.2 |
| selective_transfer[g*k] | 0.01 | 4.7 | 9.06 | True |
| selective_transfer[g*k] | 0.02 | 16.8 | 14.46 | True |
| selective_transfer[g*k] | 0.05 | 37.7 | 26.84 | True |
| selective_transfer[g*k] | 0.1 | 62.9 | 40.33 | True |
| absent_conformal[g] | 0.01 | present accepted 10.1 | false-accept absent 0.96 | |
| absent_conformal[g] | 0.02 | present accepted 14.2 | false-accept absent 1.90 | |
| absent_conformal[g] | 0.05 | present accepted 28.3 | false-accept absent 4.96 | |
| absent_conformal[g] | 0.1 | present accepted 43.1 | false-accept absent 9.66 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 41.0 | false-accept absent 8.90 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 59.4 | false-accept absent 23.03 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 77.8 | false-accept absent 50.06 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 86.3 | false-accept absent 65.63 | |

AURC ĝ 37.07 vs ĝκ̂ 32.61; paired diff -4.46 [-5.49, -3.60]

## Prop. 1 two-knob plug-in rule vs one-knob thresholds (expected cost per query, lower is better)

| set | costs | two-knob plug-in | ĝ≥.5 | ĝ thr tuned on calib | ĝκ̂ thr tuned | ĝ oracle thr | paired two−one(tuned) |
|---|---|---|---|---|---|---|---|
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0647 | 0.0622 | 0.0683 | 0.0628 | 0.0599 | -0.0035 [-0.0054, -0.0016] |
| refcoco_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1005 | 0.1193 | 0.1236 | 0.0992 | 0.1187 | -0.0231 [-0.0295, -0.0175] |
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.0691 | 0.0672 | 0.0716 | 0.0665 | 0.0606 | -0.0025 [-0.0037, -0.0015] |
| refcoco_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0691 | 0.0622 | 0.0703 | 0.0663 | 0.0599 | -0.0012 [-0.0031, 0.0005] |
| refcoco_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1400 | 0.2336 | 0.2259 | 0.1454 | 0.2259 | -0.0859 [-0.1013, -0.0714] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0926 | 0.0891 | 0.0986 | 0.0885 | 0.0863 | -0.0060 [-0.0083, -0.0038] |
| refcocog_test | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1342 | 0.1683 | 0.1737 | 0.1329 | 0.1681 | -0.0395 [-0.0472, -0.0324] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1048 | 0.0991 | 0.1079 | 0.0983 | 0.0874 | -0.0031 [-0.0045, -0.0018] |
| refcocog_test | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0985 | 0.0891 | 0.1030 | 0.0946 | 0.0863 | -0.0045 [-0.0067, -0.0020] |
| refcocog_test | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1858 | 0.3266 | 0.2988 | 0.1934 | 0.2988 | -0.1130 [-0.1300, -0.0956] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1063 | 0.1283 | 0.1113 | 0.1113 | 0.1080 | -0.0050 [-0.0071, -0.0030] |
| grec_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1327 | 0.1983 | 0.1742 | 0.1333 | 0.1661 | -0.0414 [-0.0475, -0.0364] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1302 | 0.1343 | 0.1282 | 0.1419 | 0.1271 | 0.0020 [0.0008, 0.0032] |
| grec_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1273 | 0.1807 | 0.1368 | 0.1392 | 0.1247 | -0.0095 [-0.0118, -0.0070] |
| grec_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1575 | 0.3381 | 0.2800 | 0.1561 | 0.2800 | -0.1225 [-0.1372, -0.1116] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1778 | 0.1994 | 0.1876 | 0.1800 | 0.1858 | -0.0098 [-0.0137, -0.0064] |
| grec_testA | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2168 | 0.3315 | 0.3053 | 0.2197 | 0.2905 | -0.0885 [-0.0998, -0.0779] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2160 | 0.2160 | 0.2150 | 0.2247 | 0.2144 | 0.0010 [-0.0003, 0.0024] |
| grec_testA | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2053 | 0.2500 | 0.2171 | 0.2123 | 0.2091 | -0.0118 [-0.0161, -0.0081] |
| grec_testA | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2395 | 0.5956 | 0.4966 | 0.2372 | 0.4966 | -0.2571 [-0.2792, -0.2331] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2183 | 0.2485 | 0.2309 | 0.2217 | 0.2263 | -0.0126 [-0.0167, -0.0082] |
| grec_testB | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2523 | 0.3923 | 0.3616 | 0.2556 | 0.3411 | -0.1093 [-0.1211, -0.0960] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2798 | 0.2803 | 0.2764 | 0.2968 | 0.2755 | 0.0034 [0.0017, 0.0051] |
| grec_testB | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2573 | 0.3212 | 0.2772 | 0.2684 | 0.2608 | -0.0199 [-0.0243, -0.0150] |
| grec_testB | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2733 | 0.6801 | 0.5707 | 0.2689 | 0.5707 | -0.2974 [-0.3250, -0.2695] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1089 | 0.0979 | 0.1061 | 0.1057 | 0.0892 | 0.0028 [-0.0042, 0.0098] |
| ss2 | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1675 | 0.1678 | 0.1926 | 0.1792 | 0.1616 | -0.0252 [-0.0403, -0.0107] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1234 | 0.1258 | 0.1226 | 0.1116 | 0.0920 | 0.0008 [-0.0048, 0.0062] |
| ss2 | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1230 | 0.0979 | 0.1317 | 0.1152 | 0.0892 | -0.0086 [-0.0155, -0.0017] |
| ss2 | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2327 | 0.3078 | 0.2586 | 0.2488 | 0.2512 | -0.0259 [-0.0533, 0.0033] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.3658 | 0.4370 | 0.3997 | 0.3745 | 0.3580 | -0.0339 [-0.0393, -0.0282] |
| venus | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.4496 | 0.7027 | 0.4652 | 0.4388 | 0.4652 | -0.0156 [-0.0260, -0.0038] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.5429 | 0.5426 | 0.5437 | 0.5421 | 0.5416 | -0.0008 [-0.0046, 0.0024] |
| venus | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.3873 | 0.5027 | 0.3972 | 0.4065 | 0.3732 | -0.0100 [-0.0155, -0.0038] |
| venus | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.5418 | 1.2341 | 0.6796 | 0.5051 | 0.6796 | -0.1379 [-0.1646, -0.1120] |
