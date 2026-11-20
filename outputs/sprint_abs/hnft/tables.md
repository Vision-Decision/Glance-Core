# sprint_abs tables (hnft)

## GREC official protocol (gRefCOCO; Pr@(F1=1, GIoU>=0.5), N-acc, T-acc)

| split | variant | Pr@F1=1 | 95% CI | N-acc | T-acc | Pr single | Pr multi | Pr (plain IoU) |
|---|---|---|---|---|---|---|---|---|
| grec_val | single_raw@0.7(official) | 71.4 | 71.4 [70.9, 71.9] | 89.6 | 96.6 | 91.6 | 0.0 | 71.7 |
| grec_val | single_raw@0.5 | 70.4 | 70.4 [69.9, 70.9] | 85.8 | 97.9 | 92.3 | 0.0 | 70.7 |
| grec_val | single_platt@0.7 | 71.5 | 71.5 [71.0, 72.0] | 90.1 | 96.3 | 91.4 | 0.0 | 71.8 |
| grec_val | multi_platt@0.7(official) | 71.5 | 71.5 [71.0, 72.0] | 90.1 | 96.3 | 91.4 | 0.0 | 71.8 |
| grec_val | multi_raw@0.7(official) | 71.4 | 71.4 [70.9, 71.9] | 89.6 | 96.6 | 91.6 | 0.0 | 71.7 |
| grec_val | multi_platt@tuned | 70.5 | 70.5 [69.9, 70.9] | 85.5 | 98.0 | 92.0 | 1.4 | 70.9 |
| grec_val | single_platt@tuned | 71.1 | 71.1 [70.6, 71.6] | 88.2 | 97.2 | 91.9 | 0.0 | 71.4 |
| grec_val | paired diff multi-single (official) | 0.1 [0.0, 0.2] | | | | | | |
| grec_testA | single_raw@0.7(official) | 47.4 | 47.4 [46.7, 48.2] | 83.7 | 93.2 | 91.1 | 0.0 | 47.5 |
| grec_testA | single_raw@0.5 | 46.7 | 46.7 [46.0, 47.4] | 79.7 | 95.2 | 91.6 | 0.0 | 46.8 |
| grec_testA | single_platt@0.7 | 47.5 | 47.5 [46.8, 48.2] | 84.1 | 92.8 | 91.0 | 0.0 | 47.6 |
| grec_testA | multi_platt@0.7(official) | 47.5 | 47.5 [46.8, 48.2] | 84.1 | 92.8 | 91.0 | 0.0 | 47.6 |
| grec_testA | multi_raw@0.7(official) | 47.4 | 47.4 [46.7, 48.2] | 83.7 | 93.2 | 91.1 | 0.0 | 47.5 |
| grec_testA | multi_platt@tuned | 47.1 | 47.1 [46.3, 47.8] | 79.4 | 95.4 | 91.4 | 1.1 | 47.2 |
| grec_testA | single_platt@tuned | 47.2 | 47.2 [46.5, 47.9] | 82.3 | 94.1 | 91.3 | 0.0 | 47.3 |
| grec_testA | paired diff multi-single (official) | 0.1 [0.0, 0.1] | | | | | | |
| grec_testB | single_raw@0.7(official) | 51.1 | 51.1 [50.1, 52.3] | 79.3 | 89.0 | 79.9 | 0.0 | 51.5 |
| grec_testB | single_raw@0.5 | 50.1 | 50.1 [49.1, 51.2] | 75.0 | 90.8 | 80.6 | 0.0 | 50.5 |
| grec_testB | single_platt@0.7 | 51.3 | 51.3 [50.2, 52.4] | 79.9 | 88.6 | 79.7 | 0.0 | 51.6 |
| grec_testB | multi_platt@0.7(official) | 51.3 | 51.3 [50.2, 52.4] | 79.9 | 88.6 | 79.7 | 0.0 | 51.6 |
| grec_testB | multi_raw@0.7(official) | 51.1 | 51.1 [50.1, 52.3] | 79.3 | 89.0 | 79.9 | 0.0 | 51.5 |
| grec_testB | multi_platt@tuned | 50.2 | 50.2 [49.2, 51.4] | 74.6 | 90.9 | 80.4 | 0.9 | 50.6 |
| grec_testB | single_platt@tuned | 50.8 | 50.8 [49.7, 51.9] | 77.8 | 89.6 | 80.1 | 0.0 | 51.1 |
| grec_testB | paired diff multi-single (official) | 0.1 [0.1, 0.2] | | | | | | |
| grefcalib | single_raw@0.7(official) | 63.1 | - | 96.9 | 97.5 | 91.2 | 0.0 | 63.4 |
| grefcalib | single_raw@0.5 | 63.3 | - | 95.8 | 98.5 | 91.6 | 0.0 | 63.6 |
| grefcalib | single_platt@0.7 | 63.1 | - | 97.1 | 97.3 | 91.0 | 0.0 | 63.3 |
| grefcalib | multi_platt@0.7(official) | 63.1 | - | 97.1 | 97.3 | 91.0 | 0.0 | 63.3 |
| grefcalib | multi_raw@0.7(official) | 63.1 | - | 96.9 | 97.5 | 91.2 | 0.0 | 63.4 |
| grefcalib | multi_platt@tuned | 63.5 | - | 95.8 | 98.6 | 91.2 | 1.7 | 63.9 |
| grefcalib | single_platt@tuned | 63.3 | - | 96.5 | 98.0 | 91.5 | 0.0 | 63.5 |

Tuned on grefcalib: {"pr_f1_grefcalib": 0.635385671778439, "t_presence": 0.44999999999999996, "t_extra": 0.2, "single_t_presence": 0.5999999999999999, "single_pr_f1_grefcalib": 0.6326390478370336}; extra-box model: {"n": 1455, "pos": 0.11958762886597939}

## Presence ĝ calibration (fit on held-out training-distribution splits)

| set | subset | method | AUROC | N-acc@.5 | recall@.5 | ECE15 | ECE15-mass | ECE10 | Brier | NLL | +EM prior: ECE15 / N-acc / π̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grec_val | all | raw | 98.0 | 85.8 | 97.9 | 3.4 | 3.4 | 3.4 | 4.96 | 0.184 | - |
| grec_val | all | temp | 98.0 | 85.8 | 97.9 | 3.4 | 3.4 | 3.4 | 4.96 | 0.183 | 2.7 / 86.9 / 0.670 |
| grec_val | all | platt | 98.0 | 86.3 | 97.8 | 3.1 | 3.1 | 3.1 | 4.88 | 0.181 | 2.4 / 87.7 / 0.667 |
| grec_val | all | beta | 98.0 | 86.8 | 97.6 | 3.0 | 3.0 | 3.0 | 4.84 | 0.181 | 2.3 / 88.1 / 0.666 |
| grec_val | all | iso | 98.0 | 88.1 | 97.2 | 2.9 | 2.9 | 2.9 | 4.83 | 0.181 | 2.3 / 88.7 / 0.665 |
| grec_val | all | multi | 98.0 | 85.7 | 97.9 | 3.1 | 3.1 | 3.1 | 4.86 | 0.173 | 2.0 / 87.4 / 0.665 |
| grec_val | gref_only | raw | 96.6 | 85.8 | 95.9 | 7.4 | 7.5 | 7.4 | 8.09 | 0.299 | - |
| grec_val | gref_only | temp | 96.6 | 85.8 | 95.9 | 7.4 | 7.5 | 7.4 | 8.08 | 0.298 | 2.7 / 91.7 / 0.385 |
| grec_val | gref_only | platt | 96.6 | 86.3 | 95.6 | 6.9 | 7.0 | 6.9 | 7.89 | 0.291 | 2.6 / 92.1 / 0.380 |
| grec_val | gref_only | beta | 96.6 | 86.8 | 95.4 | 6.8 | 6.8 | 6.8 | 7.79 | 0.292 | 2.8 / 92.3 / 0.379 |
| grec_val | gref_only | iso | 96.6 | 88.1 | 94.7 | 6.7 | 6.8 | 6.7 | 7.77 | 0.292 | 2.8 / 92.0 / 0.379 |
| grec_val | gref_only | multi | 96.7 | 85.7 | 96.0 | 7.5 | 7.5 | 7.5 | 7.88 | 0.275 | 1.7 / 92.7 / 0.367 |
| grec_testA | all | raw | 94.2 | 79.7 | 95.2 | 4.4 | 4.3 | 4.4 | 6.81 | 0.300 | - |
| grec_testA | all | temp | 94.2 | 79.7 | 95.2 | 4.4 | 4.3 | 4.4 | 6.81 | 0.299 | 4.4 / 78.3 / 0.777 |
| grec_testA | all | platt | 94.2 | 80.4 | 95.0 | 4.5 | 4.4 | 4.5 | 6.85 | 0.302 | 4.4 / 79.1 / 0.773 |
| grec_testA | all | beta | 94.2 | 80.8 | 94.8 | 4.6 | 4.6 | 4.6 | 6.86 | 0.298 | 4.4 / 79.5 / 0.773 |
| grec_testA | all | iso | 94.2 | 82.2 | 94.1 | 4.7 | 4.6 | 4.7 | 6.90 | 0.296 | 4.6 / 81.0 / 0.772 |
| grec_testA | all | multi | 94.2 | 79.7 | 95.3 | 3.5 | 3.4 | 3.4 | 6.66 | 0.269 | 3.5 / 77.9 / 0.776 |
| grec_testA | gref_only | raw | 93.3 | 79.7 | 94.4 | 5.2 | 5.1 | 5.2 | 8.44 | 0.336 | - |
| grec_testA | gref_only | temp | 93.3 | 79.7 | 94.4 | 5.2 | 5.1 | 5.2 | 8.43 | 0.334 | 5.3 / 80.8 / 0.676 |
| grec_testA | gref_only | platt | 93.3 | 80.4 | 94.0 | 5.3 | 5.2 | 5.3 | 8.46 | 0.335 | 5.5 / 81.6 / 0.670 |
| grec_testA | gref_only | beta | 93.3 | 80.8 | 93.8 | 5.3 | 5.3 | 5.3 | 8.46 | 0.337 | 5.6 / 82.1 / 0.669 |
| grec_testA | gref_only | iso | 93.3 | 82.2 | 92.8 | 5.6 | 5.6 | 5.6 | 8.52 | 0.338 | 5.7 / 82.7 / 0.668 |
| grec_testA | gref_only | multi | 93.5 | 79.7 | 94.6 | 4.1 | 3.9 | 4.0 | 8.23 | 0.304 | 4.2 / 81.2 / 0.670 |
| grec_testB | all | raw | 88.9 | 75.0 | 90.8 | 9.4 | 9.2 | 9.4 | 11.74 | 0.622 | - |
| grec_testB | all | temp | 88.9 | 75.0 | 90.8 | 9.4 | 9.2 | 9.4 | 11.72 | 0.617 | 9.2 / 75.4 / 0.703 |
| grec_testB | all | platt | 88.9 | 75.6 | 90.6 | 9.3 | 9.1 | 9.3 | 11.73 | 0.626 | 9.2 / 76.3 / 0.698 |
| grec_testB | all | beta | 88.9 | 76.2 | 90.4 | 9.3 | 8.9 | 9.2 | 11.71 | 0.602 | 9.2 / 76.8 / 0.697 |
| grec_testB | all | iso | 88.9 | 77.7 | 89.7 | 9.5 | 8.8 | 9.5 | 11.75 | 0.589 | 9.4 / 78.0 / 0.697 |
| grec_testB | all | multi | 89.0 | 74.8 | 90.9 | 8.2 | 8.2 | 8.3 | 11.49 | 0.538 | 8.1 / 75.7 / 0.696 |
| grec_testB | gref_only | raw | 90.2 | 75.0 | 92.3 | 9.6 | 9.2 | 9.6 | 12.41 | 0.500 | - |
| grec_testB | gref_only | temp | 90.2 | 75.0 | 92.3 | 9.5 | 9.1 | 9.5 | 12.39 | 0.497 | 8.6 / 78.4 / 0.582 |
| grec_testB | gref_only | platt | 90.2 | 75.6 | 92.1 | 9.4 | 9.1 | 9.4 | 12.35 | 0.495 | 8.5 / 78.9 / 0.576 |
| grec_testB | gref_only | beta | 90.2 | 76.2 | 91.8 | 9.3 | 9.0 | 9.3 | 12.31 | 0.499 | 8.6 / 79.4 / 0.574 |
| grec_testB | gref_only | iso | 90.2 | 77.7 | 90.6 | 9.4 | 9.4 | 9.4 | 12.36 | 0.500 | 8.9 / 78.8 / 0.573 |
| grec_testB | gref_only | multi | 90.3 | 74.8 | 92.5 | 8.3 | 8.1 | 8.3 | 12.03 | 0.443 | 6.9 / 79.3 / 0.569 |
| calib_natural | all | raw | 99.4 | 94.3 | 98.8 | 0.5 | 0.3 | 0.3 | 1.91 | 0.072 | - |
| calib_natural | all | temp | 99.4 | 94.3 | 98.8 | 0.5 | 0.3 | 0.3 | 1.91 | 0.072 | 0.5 / 94.3 / 0.725 |
| calib_natural | all | platt | 99.4 | 94.5 | 98.7 | 0.4 | 0.1 | 0.4 | 1.90 | 0.072 | 0.4 / 94.5 / 0.723 |
| calib_natural | all | beta | 99.4 | 94.7 | 98.7 | 0.4 | 0.1 | 0.4 | 1.90 | 0.072 | 0.4 / 94.7 / 0.723 |
| calib_natural | all | iso | 99.4 | 95.6 | 98.5 | 0.3 | 0.1 | 0.1 | 1.88 | 0.071 | 0.3 / 95.6 / 0.723 |
| calib_natural | all | multi | 99.4 | 94.4 | 98.7 | 1.0 | 0.8 | 1.0 | 1.91 | 0.074 | 1.0 / 94.4 / 0.721 |
| grefcalib | all | raw | 99.4 | 95.8 | 98.5 | 1.9 | 2.0 | 1.9 | 1.38 | 0.054 | - |
| grefcalib | all | temp | 99.4 | 95.8 | 98.5 | 2.0 | 2.0 | 2.0 | 1.38 | 0.054 | 0.4 / 93.1 / 0.893 |
| grefcalib | all | platt | 99.4 | 95.8 | 98.4 | 2.1 | 2.2 | 2.1 | 1.45 | 0.056 | 0.5 / 93.3 / 0.892 |
| grefcalib | all | beta | 99.4 | 95.8 | 98.3 | 2.1 | 2.2 | 2.1 | 1.48 | 0.056 | 0.5 / 93.8 / 0.892 |
| grefcalib | all | iso | 99.4 | 96.5 | 98.1 | 2.2 | 2.2 | 2.2 | 1.51 | 0.056 | 0.6 / 93.3 / 0.891 |
| grefcalib | all | multi | 99.4 | 95.8 | 98.6 | 2.5 | 2.5 | 2.5 | 1.39 | 0.057 | 0.8 / 92.5 / 0.893 |
| guisyn_test | natural_fit | raw | 99.3 | 95.2 | 97.6 | 0.9 | 0.7 | 0.9 | 2.41 | 0.091 | - |
| guisyn_test | natural_fit | temp | 99.3 | 95.2 | 97.6 | 0.9 | 0.7 | 0.9 | 2.41 | 0.091 | 0.8 / 94.9 / 0.746 |
| guisyn_test | natural_fit | platt | 99.3 | 95.6 | 97.5 | 0.9 | 0.8 | 0.9 | 2.42 | 0.092 | 0.9 / 95.2 / 0.743 |
| guisyn_test | natural_fit | beta | 99.3 | 95.8 | 97.4 | 0.9 | 0.8 | 0.9 | 2.42 | 0.091 | 0.9 / 95.5 / 0.743 |
| guisyn_test | natural_fit | iso | 99.3 | 96.5 | 97.1 | 0.9 | 0.9 | 0.9 | 2.43 | 0.091 | 0.9 / 96.3 / 0.743 |
| guisyn_test | natural_fit | multi | 99.2 | 93.0 | 97.9 | 1.3 | 1.3 | 1.3 | 2.62 | 0.094 | 1.8 / 91.6 / 0.765 |
| guisyn_test | gui_fit | raw | 99.3 | 95.2 | 97.6 | 0.9 | 0.7 | 0.9 | 2.41 | 0.091 | - |
| guisyn_test | gui_fit | temp | 99.3 | 95.2 | 97.6 | 0.6 | 0.4 | 0.5 | 2.40 | 0.088 | 0.7 / 95.6 / 0.742 |
| guisyn_test | gui_fit | platt | 99.3 | 94.7 | 97.7 | 0.5 | 0.4 | 0.4 | 2.40 | 0.088 | 0.5 / 95.0 / 0.745 |
| guisyn_test | gui_fit | beta | 99.3 | 94.7 | 97.7 | 0.5 | 0.4 | 0.4 | 2.40 | 0.088 | 0.5 / 95.0 / 0.745 |
| guisyn_test | gui_fit | iso | 99.3 | 93.8 | 97.9 | 0.5 | 0.4 | 0.5 | 2.41 | 0.088 | 0.6 / 94.4 / 0.745 |
| guisyn_test | gui_fit | multi | 99.3 | 94.9 | 97.7 | 0.6 | 0.5 | 0.5 | 2.42 | 0.090 | 0.6 / 95.1 / 0.742 |
| venus | natural_fit | raw | 71.0 | 54.6 | 73.2 | 21.3 | 21.2 | 21.2 | 22.89 | 0.829 | - |
| venus | natural_fit | temp | 71.0 | 54.6 | 73.2 | 21.2 | 21.1 | 21.1 | 22.85 | 0.824 | 23.7 / 58.8 / 0.640 |
| venus | natural_fit | platt | 71.0 | 56.2 | 72.0 | 22.1 | 22.0 | 21.8 | 23.44 | 0.855 | 25.2 / 62.4 / 0.623 |
| venus | natural_fit | beta | 71.0 | 57.2 | 71.0 | 22.5 | 22.4 | 22.3 | 23.57 | 0.837 | 25.9 / 64.5 / 0.616 |
| venus | natural_fit | iso | 71.0 | 60.6 | 68.4 | 22.6 | 22.6 | 22.4 | 23.94 | 0.848 | 26.1 / 64.4 / 0.614 |
| venus | natural_fit | multi | 72.2 | 47.3 | 78.1 | 16.3 | 16.3 | 16.4 | 18.88 | 0.608 | 17.4 / 49.6 / 0.697 |
| venus | gui_fit | raw | 71.0 | 54.6 | 73.2 | 21.3 | 21.2 | 21.2 | 22.89 | 0.829 | - |
| venus | gui_fit | temp | 71.0 | 54.6 | 73.2 | 20.6 | 20.6 | 20.4 | 22.21 | 0.755 | 27.5 / 67.9 / 0.589 |
| venus | gui_fit | platt | 71.0 | 51.4 | 74.4 | 19.6 | 19.5 | 19.3 | 21.56 | 0.730 | 25.1 / 63.4 / 0.615 |
| venus | gui_fit | beta | 71.0 | 51.4 | 74.4 | 19.6 | 19.5 | 19.3 | 21.55 | 0.731 | 25.0 / 63.4 / 0.615 |
| venus | gui_fit | iso | 71.0 | 48.6 | 76.3 | 19.2 | 19.2 | 19.0 | 21.61 | 0.733 | 25.0 / 63.7 / 0.615 |
| venus | gui_fit | multi | 69.0 | 49.4 | 72.2 | 20.0 | 20.1 | 20.0 | 22.28 | 0.714 | 28.1 / 66.1 / 0.577 |

## κ̂ calibration (natural: event IoU>=0.5; GUI: click in box)

| set | fit | method | AUROC | AURC | ECE15 | ECE15-mass | Brier | mean κ̂ | rate |
|---|---|---|---|---|---|---|---|---|---|
| refcoco_val | natural_fit | raw | 90.1 | 1.00 | 0.9 | 0.8 | 4.24 | 94.4 | 93.9 |
| refcoco_val | natural_fit | temp | 90.1 | 1.00 | 0.8 | 0.7 | 4.23 | 93.9 | 93.9 |
| refcoco_val | natural_fit | platt | 90.1 | 1.00 | 0.8 | 0.8 | 4.23 | 93.6 | 93.9 |
| refcoco_val | natural_fit | beta | 90.1 | 1.00 | 0.7 | 0.5 | 4.24 | 93.6 | 93.9 |
| refcoco_val | natural_fit | iso | 90.1 | 1.00 | 0.6 | 0.6 | 4.23 | 93.5 | 93.9 |
| refcoco_val | natural_fit | multi | 90.2 | 1.00 | 1.0 | 1.1 | 4.21 | 93.2 | 93.9 |
| refcocog_test | natural_fit | raw | 88.6 | 1.67 | 1.2 | 0.9 | 5.81 | 92.2 | 91.4 |
| refcocog_test | natural_fit | temp | 88.6 | 1.67 | 0.6 | 0.7 | 5.80 | 91.6 | 91.4 |
| refcocog_test | natural_fit | platt | 88.6 | 1.67 | 0.5 | 0.9 | 5.81 | 91.1 | 91.4 |
| refcocog_test | natural_fit | beta | 88.6 | 1.67 | 0.8 | 0.9 | 5.82 | 91.0 | 91.4 |
| refcocog_test | natural_fit | iso | 88.6 | 1.67 | 0.9 | 1.2 | 5.81 | 90.8 | 91.4 |
| refcocog_test | natural_fit | multi | 88.7 | 1.66 | 0.7 | 0.8 | 5.79 | 90.8 | 91.4 |
| grec_val | natural_fit | raw | 90.3 | 0.99 | 0.8 | 0.8 | 4.23 | 94.4 | 93.8 |
| grec_val | natural_fit | temp | 90.3 | 0.99 | 0.8 | 0.7 | 4.22 | 93.9 | 93.8 |
| grec_val | natural_fit | platt | 90.3 | 0.99 | 0.8 | 0.8 | 4.23 | 93.6 | 93.8 |
| grec_val | natural_fit | beta | 90.3 | 0.99 | 0.6 | 0.5 | 4.24 | 93.6 | 93.8 |
| grec_val | natural_fit | iso | 90.3 | 0.99 | 0.4 | 0.6 | 4.23 | 93.5 | 93.8 |
| grec_val | natural_fit | multi | 90.4 | 0.99 | 0.9 | 0.9 | 4.20 | 93.2 | 93.8 |
| grec_testA | natural_fit | raw | 92.8 | 0.89 | 1.5 | 1.7 | 3.88 | 94.6 | 93.1 |
| grec_testA | natural_fit | temp | 92.8 | 0.89 | 1.2 | 1.6 | 3.86 | 94.2 | 93.1 |
| grec_testA | natural_fit | platt | 92.8 | 0.89 | 1.0 | 1.2 | 3.81 | 93.9 | 93.1 |
| grec_testA | natural_fit | beta | 92.8 | 0.89 | 1.0 | 1.2 | 3.84 | 94.0 | 93.1 |
| grec_testA | natural_fit | iso | 92.8 | 0.89 | 1.1 | 1.0 | 3.82 | 93.9 | 93.1 |
| grec_testA | natural_fit | multi | 93.1 | 0.85 | 1.3 | 1.4 | 3.82 | 93.6 | 93.1 |
| grec_testB | natural_fit | raw | 93.1 | 2.64 | 2.2 | 2.7 | 7.02 | 86.5 | 84.2 |
| grec_testB | natural_fit | temp | 93.1 | 2.64 | 2.2 | 2.6 | 7.03 | 86.0 | 84.2 |
| grec_testB | natural_fit | platt | 93.1 | 2.64 | 1.6 | 1.7 | 6.91 | 85.1 | 84.2 |
| grec_testB | natural_fit | beta | 93.1 | 2.64 | 1.9 | 2.3 | 7.05 | 85.5 | 84.2 |
| grec_testB | natural_fit | iso | 93.1 | 2.64 | 1.6 | 2.0 | 6.94 | 84.8 | 84.2 |
| grec_testB | natural_fit | multi | 93.0 | 2.65 | 2.1 | 1.9 | 6.99 | 84.9 | 84.2 |
| calib_natural | natural_fit | raw | 91.0 | 1.26 | 1.4 | 1.4 | 5.43 | 92.6 | 91.6 |
| calib_natural | natural_fit | temp | 91.0 | 1.26 | 1.3 | 1.2 | 5.41 | 92.1 | 91.6 |
| calib_natural | natural_fit | platt | 91.0 | 1.26 | 1.1 | 1.1 | 5.40 | 91.6 | 91.6 |
| calib_natural | natural_fit | beta | 91.0 | 1.26 | 0.9 | 0.9 | 5.40 | 91.6 | 91.6 |
| calib_natural | natural_fit | iso | 91.0 | 1.26 | 0.4 | 0.5 | 5.36 | 91.4 | 91.6 |
| calib_natural | natural_fit | multi | 91.2 | 1.23 | 1.1 | 0.9 | 5.37 | 91.3 | 91.6 |
| guisyn_test | natural_fit | raw | 88.0 | 3.13 | 21.6 | 21.6 | 14.13 | 64.9 | 86.5 |
| guisyn_test | natural_fit | temp | 88.0 | 3.13 | 22.2 | 22.2 | 14.20 | 64.3 | 86.5 |
| guisyn_test | natural_fit | platt | 88.0 | 3.13 | 24.6 | 24.6 | 15.73 | 62.0 | 86.5 |
| guisyn_test | natural_fit | beta | 88.0 | 3.13 | 23.6 | 23.6 | 14.94 | 62.9 | 86.5 |
| guisyn_test | natural_fit | iso | 87.8 | 3.19 | 25.9 | 25.9 | 16.22 | 60.6 | 86.5 |
| guisyn_test | natural_fit | multi | 88.3 | 3.06 | 72.4 | 72.4 | 63.73 | 14.1 | 86.5 |
| guisyn_test | gui_fit | raw | 88.0 | 3.13 | 21.6 | 21.6 | 14.13 | 64.9 | 86.5 |
| guisyn_test | gui_fit | temp | 88.0 | 3.13 | 20.6 | 20.6 | 14.08 | 65.9 | 86.5 |
| guisyn_test | gui_fit | platt | 88.0 | 3.13 | 1.0 | 0.9 | 7.63 | 86.8 | 86.5 |
| guisyn_test | gui_fit | beta | 88.0 | 3.13 | 1.0 | 0.8 | 7.63 | 86.9 | 86.5 |
| guisyn_test | gui_fit | iso | 88.0 | 3.13 | 1.2 | 1.1 | 7.65 | 86.7 | 86.5 |
| guisyn_test | gui_fit | multi | 88.9 | 2.95 | 0.9 | 0.6 | 7.36 | 86.9 | 86.5 |
| ss2 | natural_fit | raw | 78.7 | 2.85 | 25.3 | 25.3 | 17.44 | 66.1 | 91.4 |
| ss2 | natural_fit | temp | 78.7 | 2.85 | 25.7 | 25.7 | 17.31 | 65.6 | 91.4 |
| ss2 | natural_fit | platt | 78.7 | 2.85 | 28.1 | 28.1 | 19.22 | 63.2 | 91.4 |
| ss2 | natural_fit | beta | 78.7 | 2.85 | 27.0 | 27.0 | 17.43 | 64.4 | 91.4 |
| ss2 | natural_fit | iso | 78.6 | 2.87 | 29.1 | 29.1 | 19.73 | 62.3 | 91.4 |
| ss2 | natural_fit | multi | 81.9 | 2.45 | 75.1 | 75.1 | 65.73 | 16.2 | 91.4 |
| ss2 | gui_fit | raw | 78.7 | 2.85 | 25.3 | 25.3 | 17.44 | 66.1 | 91.4 |
| ss2 | gui_fit | temp | 78.7 | 2.85 | 24.5 | 24.5 | 17.73 | 66.9 | 91.4 |
| ss2 | gui_fit | platt | 78.7 | 2.85 | 4.4 | 5.2 | 7.86 | 88.6 | 91.4 |
| ss2 | gui_fit | beta | 78.7 | 2.85 | 4.5 | 5.2 | 7.87 | 88.5 | 91.4 |
| ss2 | gui_fit | iso | 78.7 | 2.86 | 5.2 | 5.6 | 7.96 | 88.2 | 91.4 |
| ss2 | gui_fit | multi | 82.8 | 2.36 | 4.7 | 5.0 | 7.20 | 87.2 | 91.4 |
| venus | natural_fit | raw | 82.7 | 26.42 | 9.5 | 9.7 | 18.22 | 40.3 | 49.4 |
| venus | natural_fit | temp | 82.7 | 26.42 | 9.1 | 9.2 | 18.07 | 40.5 | 49.4 |
| venus | natural_fit | platt | 82.7 | 26.42 | 12.0 | 11.9 | 18.84 | 37.7 | 49.4 |
| venus | natural_fit | beta | 82.7 | 26.42 | 9.0 | 8.7 | 18.00 | 42.6 | 49.4 |
| venus | natural_fit | iso | 82.3 | 26.94 | 13.8 | 13.8 | 19.43 | 35.7 | 49.4 |
| venus | natural_fit | multi | 82.8 | 26.40 | 43.1 | 43.1 | 40.47 | 6.3 | 49.4 |
| venus | gui_fit | raw | 82.7 | 26.42 | 9.5 | 9.7 | 18.22 | 40.3 | 49.4 |
| venus | gui_fit | temp | 82.7 | 26.42 | 10.5 | 10.7 | 18.54 | 39.9 | 49.4 |
| venus | gui_fit | platt | 82.7 | 26.42 | 17.7 | 17.7 | 20.65 | 67.1 | 49.4 |
| venus | gui_fit | beta | 82.7 | 26.42 | 17.9 | 17.9 | 20.62 | 67.3 | 49.4 |
| venus | gui_fit | iso | 82.7 | 26.42 | 17.8 | 17.8 | 20.56 | 67.2 | 49.4 |
| venus | gui_fit | multi | 83.2 | 26.26 | 13.9 | 13.8 | 19.39 | 62.9 | 49.4 |

## Box confidence ĝ·κ̂ vs (present ∧ IoU>=0.5)  [the v0→v1 regression metric]

| set | method | ECE15 | ECE15-mass | Brier | π̂ (EM) |
|---|---|---|---|---|---|
| refcoco_val | raw | 1.35 | 1.30 | 4.79 | - |
| refcoco_val | evalpy_T | 1.34 | 1.28 | 4.79 | - |
| refcoco_val | g:platt|k:platt | 2.00 | 1.95 | 4.86 | - |
| refcoco_val | g:platt+em|k:platt | 0.83 | 0.90 | 4.24 | 0.999 |
| refcoco_val | g:platt|k:iso | 2.02 | 2.15 | 4.87 | - |
| refcoco_val | g:platt+em|k:iso | 0.62 | 0.71 | 4.24 | 0.999 |
| refcoco_val | g:platt|k:multi | 2.40 | 2.32 | 4.84 | - |
| refcoco_val | g:platt+em|k:multi | 1.04 | 1.02 | 4.22 | 0.999 |
| refcoco_val | g:iso|k:platt | 1.96 | 1.96 | 4.90 | - |
| refcoco_val | g:iso+em|k:platt | 0.83 | 0.91 | 4.24 | 0.999 |
| refcoco_val | g:iso|k:iso | 2.05 | 2.14 | 4.90 | - |
| refcoco_val | g:iso+em|k:iso | 0.62 | 0.70 | 4.24 | 0.999 |
| refcoco_val | g:iso|k:multi | 2.33 | 2.33 | 4.88 | - |
| refcoco_val | g:iso+em|k:multi | 1.03 | 1.02 | 4.22 | 0.999 |
| refcoco_val | kappa_only_raw | 0.89 | 0.80 | 4.24 | - |
| refcocog_test | raw | 2.06 | 2.32 | 6.51 | - |
| refcocog_test | evalpy_T | 2.11 | 2.38 | 6.51 | - |
| refcocog_test | g:platt|k:platt | 3.19 | 3.28 | 6.65 | - |
| refcocog_test | g:platt+em|k:platt | 0.56 | 0.86 | 5.81 | 0.999 |
| refcocog_test | g:platt|k:iso | 3.48 | 3.64 | 6.68 | - |
| refcocog_test | g:platt+em|k:iso | 0.88 | 1.22 | 5.82 | 0.999 |
| refcocog_test | g:platt|k:multi | 3.44 | 3.49 | 6.64 | - |
| refcocog_test | g:platt+em|k:multi | 0.80 | 0.81 | 5.80 | 0.999 |
| refcocog_test | g:iso|k:platt | 3.24 | 3.31 | 6.72 | - |
| refcocog_test | g:iso+em|k:platt | 0.55 | 0.84 | 5.81 | 0.999 |
| refcocog_test | g:iso|k:iso | 3.53 | 3.68 | 6.76 | - |
| refcocog_test | g:iso+em|k:iso | 0.91 | 1.22 | 5.82 | 0.999 |
| refcocog_test | g:iso|k:multi | 3.49 | 3.51 | 6.72 | - |
| refcocog_test | g:iso+em|k:multi | 0.79 | 0.81 | 5.80 | 0.999 |
| refcocog_test | kappa_only_raw | 1.19 | 0.92 | 5.81 | - |
| grec_val | raw | 5.35 | 5.35 | 6.39 | - |
| grec_val | evalpy_T | 5.34 | 5.34 | 6.38 | - |
| grec_val | g:platt|k:platt | 4.51 | 4.51 | 6.17 | - |
| grec_val | g:platt+em|k:platt | 3.19 | 3.21 | 5.71 | 0.589 |
| grec_val | g:platt|k:iso | 4.45 | 4.45 | 6.14 | - |
| grec_val | g:platt+em|k:iso | 3.14 | 3.14 | 5.69 | 0.589 |
| grec_val | g:platt|k:multi | 4.29 | 4.29 | 6.13 | - |
| grec_val | g:platt+em|k:multi | 2.98 | 3.00 | 5.68 | 0.589 |
| grec_val | g:iso|k:platt | 4.41 | 4.41 | 6.10 | - |
| grec_val | g:iso+em|k:platt | 3.10 | 3.10 | 5.68 | 0.588 |
| grec_val | g:iso|k:iso | 4.34 | 4.34 | 6.07 | - |
| grec_val | g:iso+em|k:iso | 3.05 | 3.05 | 5.66 | 0.588 |
| grec_val | g:iso|k:multi | 4.19 | 4.19 | 6.06 | - |
| grec_val | g:iso+em|k:multi | 2.89 | 2.94 | 5.65 | 0.588 |
| grec_val | kappa_only_raw | 28.12 | 28.12 | 22.71 | - |
| grec_testA | raw | 7.68 | 7.87 | 8.25 | - |
| grec_testA | evalpy_T | 7.67 | 7.86 | 8.24 | - |
| grec_testA | g:platt|k:platt | 6.97 | 7.15 | 7.97 | - |
| grec_testA | g:platt+em|k:platt | 6.03 | 6.19 | 7.52 | 0.623 |
| grec_testA | g:platt|k:iso | 6.97 | 7.15 | 7.96 | - |
| grec_testA | g:platt+em|k:iso | 6.06 | 6.20 | 7.52 | 0.623 |
| grec_testA | g:platt|k:multi | 6.77 | 6.92 | 7.93 | - |
| grec_testA | g:platt+em|k:multi | 5.83 | 5.97 | 7.49 | 0.623 |
| grec_testA | g:iso|k:platt | 6.80 | 6.97 | 7.88 | - |
| grec_testA | g:iso+em|k:platt | 5.81 | 6.03 | 7.47 | 0.622 |
| grec_testA | g:iso|k:iso | 6.82 | 7.00 | 7.87 | - |
| grec_testA | g:iso+em|k:iso | 5.84 | 6.06 | 7.47 | 0.622 |
| grec_testA | g:iso|k:multi | 6.60 | 6.77 | 7.84 | - |
| grec_testA | g:iso+em|k:multi | 5.61 | 5.83 | 7.44 | 0.622 |
| grec_testA | kappa_only_raw | 29.14 | 29.14 | 24.11 | - |
| grec_testB | raw | 10.06 | 10.05 | 11.76 | - |
| grec_testB | evalpy_T | 10.05 | 10.03 | 11.75 | - |
| grec_testB | g:platt|k:platt | 9.10 | 9.09 | 11.44 | - |
| grec_testB | g:platt+em|k:platt | 7.65 | 7.66 | 10.73 | 0.574 |
| grec_testB | g:platt|k:iso | 8.86 | 8.89 | 11.36 | - |
| grec_testB | g:platt+em|k:iso | 7.48 | 7.47 | 10.66 | 0.574 |
| grec_testB | g:platt|k:multi | 8.94 | 8.95 | 11.40 | - |
| grec_testB | g:platt+em|k:multi | 7.53 | 7.53 | 10.69 | 0.574 |
| grec_testB | g:iso|k:platt | 8.92 | 9.00 | 11.37 | - |
| grec_testB | g:iso+em|k:platt | 7.50 | 7.62 | 10.71 | 0.573 |
| grec_testB | g:iso|k:iso | 8.71 | 8.79 | 11.29 | - |
| grec_testB | g:iso+em|k:iso | 7.29 | 7.38 | 10.65 | 0.573 |
| grec_testB | g:iso|k:multi | 8.78 | 8.84 | 11.33 | - |
| grec_testB | g:iso+em|k:multi | 7.36 | 7.46 | 10.68 | 0.573 |
| grec_testB | kappa_only_raw | 31.29 | 31.29 | 26.98 | - |

Bootstrap (image clusters): 
- refcoco_val: raw ECE 1.35 [1.15, 1.90]; fixed 0.83 [0.64, 1.37]; paired diff -0.52 [-0.82, -0.19]; κ̂ AUROC 90.1 [88.5, 91.8]
- refcocog_test: raw ECE 2.06 [1.79, 2.66]; fixed 0.56 [0.59, 1.34]; paired diff -1.51 [-1.72, -0.88]; κ̂ AUROC 88.6 [87.4, 89.8]

## Presence bootstrap CIs (image clusters, raw ĝ unless noted)

- grec_val: AUROC 98.0 [97.8, 98.2]; N-acc 85.8 [85.2, 86.5]; ECE raw 3.45 [3.18, 3.73] → Platt 3.14 [2.87, 3.43] (paired -0.31 [-0.35, -0.26]); subset ECE raw 7.45 [7.03, 7.92] → Platt 6.92 [6.52, 7.39]
- grec_testA: AUROC 94.2 [93.6, 94.6]; N-acc 79.7 [78.5, 81.0]; ECE raw 4.40 [4.08, 5.03] → Platt 4.48 [4.12, 5.06] (paired 0.08 [-0.04, 0.18]); subset ECE raw 4.40 [4.08, 5.03] → Platt 4.48 [4.12, 5.06]
- grec_testB: AUROC 88.9 [87.6, 90.3]; N-acc 75.0 [73.9, 76.3]; ECE raw 9.42 [8.24, 10.60] → Platt 9.33 [8.18, 10.53] (paired -0.09 [-0.18, 0.06]); subset ECE raw 9.42 [8.24, 10.60] → Platt 9.33 [8.18, 10.53]
- venus: AUROC 71.0 [69.0, 72.6]; N-acc 54.6 [50.6, 57.8]; ECE raw 21.31 [20.07, 22.77] → Platt 19.64 [18.51, 21.09] (paired -1.67 [-1.92, -1.35]); subset ECE raw 21.31 [20.07, 22.77] → Platt 19.64 [18.51, 21.09]

## Risk-controlled accept/abstain (δ=0.1; 50/50 image splits ×reps; coverage = fraction answered)


**refcoco_val** (n=10834, absent 0.0%, correct-if-answered-all 93.9%); default ĝ≥.5: coverage 98.9, selective risk 5.9, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 24.7 | 0.97 | 0.11 |
| selective[g] | 0.05 | 85.2 | 4.19 | 0.12 |
| selective[g] | 0.1 | 100.0 | 6.13 | 0.0 |
| joint[g] | 0.01 | 50.4 | 0.85 | 0.26 |
| joint[g] | 0.02 | 69.1 | 1.76 | 0.22 |
| joint[g] | 0.05 | 91.2 | 4.53 | 0.23 |
| joint[g] | 0.1 | 100.0 | 6.14 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 6.1 | 0.60 | False |
| selective_transfer[g] | 0.05 | 75.9 | 3.13 | False |
| selective_transfer[g] | 0.1 | 97.6 | 5.82 | False |
| selective[g*k] | 0.01 | 29.6 | 0.41 | 0.05 |
| selective[g*k] | 0.02 | 78.7 | 1.52 | 0.09 |
| selective[g*k] | 0.05 | 95.4 | 4.21 | 0.09 |
| selective[g*k] | 0.1 | 100.0 | 6.12 | 0.0 |
| joint[g*k] | 0.01 | 72.7 | 0.81 | 0.24 |
| joint[g*k] | 0.02 | 85.1 | 1.78 | 0.24 |
| joint[g*k] | 0.05 | 96.7 | 4.51 | 0.15 |
| joint[g*k] | 0.1 | 100.0 | 6.14 | 0.0 |
| selective_transfer[g*k] | 0.01 | 52.3 | 0.69 | False |
| selective_transfer[g*k] | 0.02 | 72.2 | 1.05 | False |
| selective_transfer[g*k] | 0.05 | 89.5 | 2.77 | False |
| selective_transfer[g*k] | 0.1 | 98.0 | 5.22 | False |

AURC ĝ 2.24 vs ĝκ̂ 1.12; paired diff -1.12 [-1.40, -0.82]

**refcocog_test** (n=9602, absent 0.0%, correct-if-answered-all 91.4%); default ĝ≥.5: coverage 98.2, selective risk 8.1, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 1.3 | 0.16 | 0.02 |
| selective[g] | 0.05 | 61.6 | 4.01 | 0.06 |
| selective[g] | 0.1 | 99.9 | 8.58 | 0.0 |
| joint[g] | 0.01 | 36.3 | 0.80 | 0.15 |
| joint[g] | 0.02 | 52.4 | 1.74 | 0.2 |
| joint[g] | 0.05 | 78.3 | 4.54 | 0.17 |
| joint[g] | 0.1 | 100.0 | 8.61 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 1.5 | 2.13 | True |
| selective_transfer[g] | 0.05 | 57.1 | 3.63 | False |
| selective_transfer[g] | 0.1 | 95.9 | 7.79 | False |
| selective[g*k] | 0.01 | 6.4 | 0.15 | 0.01 |
| selective[g*k] | 0.02 | 64.3 | 1.48 | 0.06 |
| selective[g*k] | 0.05 | 87.0 | 4.21 | 0.07 |
| selective[g*k] | 0.1 | 99.9 | 8.58 | 0.0 |
| joint[g*k] | 0.01 | 62.0 | 0.79 | 0.1 |
| joint[g*k] | 0.02 | 74.4 | 1.77 | 0.25 |
| joint[g*k] | 0.05 | 90.6 | 4.55 | 0.12 |
| joint[g*k] | 0.1 | 100.0 | 8.61 | 0.0 |
| selective_transfer[g*k] | 0.01 | 39.4 | 0.63 | False |
| selective_transfer[g*k] | 0.02 | 59.5 | 1.09 | False |
| selective_transfer[g*k] | 0.05 | 84.2 | 3.71 | False |
| selective_transfer[g*k] | 0.1 | 96.6 | 7.05 | False |

AURC ĝ 3.62 vs ĝκ̂ 1.77; paired diff -1.86 [-2.19, -1.52]

**grec_val** (n=25063, absent 35.5%, correct-if-answered-all 56.3%); default ĝ≥.5: coverage 68.2, selective risk 18.7, false-accept on absent 14.2

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 40.3 | 9.01 | 0.1 |
| joint[g] | 0.01 | 15.1 | 0.87 | 0.14 |
| joint[g] | 0.02 | 27.0 | 1.81 | 0.21 |
| joint[g] | 0.05 | 45.9 | 4.68 | 0.17 |
| joint[g] | 0.1 | 62.9 | 9.58 | 0.25 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 3.0 | 7.92 | True |
| selective_transfer[g] | 0.05 | 41.5 | 9.21 | True |
| selective_transfer[g] | 0.1 | 65.0 | 16.40 | True |
| selective[g*k] | 0.01 | 0.1 | 0.02 | 0.01 |
| selective[g*k] | 0.02 | 8.0 | 0.80 | 0.02 |
| selective[g*k] | 0.05 | 40.1 | 4.30 | 0.1 |
| selective[g*k] | 0.1 | 53.2 | 9.17 | 0.05 |
| joint[g*k] | 0.01 | 31.9 | 0.89 | 0.18 |
| joint[g*k] | 0.02 | 40.8 | 1.81 | 0.18 |
| joint[g*k] | 0.05 | 52.7 | 4.67 | 0.18 |
| joint[g*k] | 0.1 | 63.1 | 9.48 | 0.07 |
| selective_transfer[g*k] | 0.01 | 24.3 | 2.10 | True |
| selective_transfer[g*k] | 0.02 | 35.4 | 3.22 | True |
| selective_transfer[g*k] | 0.05 | 50.1 | 7.66 | True |
| selective_transfer[g*k] | 0.1 | 65.7 | 16.82 | True |
| absent_conformal[g] | 0.01 | present accepted 59.0 | false-accept absent 1.01 | |
| absent_conformal[g] | 0.02 | present accepted 73.7 | false-accept absent 2.04 | |
| absent_conformal[g] | 0.05 | present accepted 90.4 | false-accept absent 5.12 | |
| absent_conformal[g] | 0.1 | present accepted 96.4 | false-accept absent 10.05 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 86.1 | false-accept absent 3.68 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 92.2 | false-accept absent 5.93 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 97.5 | false-accept absent 12.75 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 99.2 | false-accept absent 22.17 | |

AURC ĝ 16.76 vs ĝκ̂ 13.94; paired diff -2.82 [-3.24, -2.48]

**grec_testA** (n=19200, absent 23.2%, correct-if-answered-all 59.9%); default ĝ≥.5: coverage 77.8, selective risk 24.8, false-accept on absent 20.3

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 3.0 | 0.86 | 0.31 |
| joint[g] | 0.02 | 9.7 | 1.79 | 0.28 |
| joint[g] | 0.05 | 29.2 | 4.68 | 0.24 |
| joint[g] | 0.1 | 52.7 | 9.65 | 0.38 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 4.3 | 24.61 | True |
| selective_transfer[g] | 0.05 | 46.5 | 17.47 | True |
| selective_transfer[g] | 0.1 | 74.3 | 23.33 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 21.8 | 3.58 | 0.03 |
| selective[g*k] | 0.1 | 46.1 | 9.03 | 0.11 |
| joint[g*k] | 0.01 | 22.9 | 0.86 | 0.21 |
| joint[g*k] | 0.02 | 33.7 | 1.82 | 0.19 |
| joint[g*k] | 0.05 | 47.9 | 4.61 | 0.15 |
| joint[g*k] | 0.1 | 60.5 | 9.58 | 0.29 |
| selective_transfer[g*k] | 0.01 | 22.9 | 3.71 | True |
| selective_transfer[g*k] | 0.02 | 32.3 | 5.09 | True |
| selective_transfer[g*k] | 0.05 | 49.3 | 10.08 | True |
| selective_transfer[g*k] | 0.1 | 73.7 | 22.42 | True |
| absent_conformal[g] | 0.01 | present accepted 22.7 | false-accept absent 0.99 | |
| absent_conformal[g] | 0.02 | present accepted 35.6 | false-accept absent 1.96 | |
| absent_conformal[g] | 0.05 | present accepted 66.9 | false-accept absent 4.81 | |
| absent_conformal[g] | 0.1 | present accepted 85.1 | false-accept absent 10.10 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 80.4 | false-accept absent 8.07 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 87.7 | false-accept absent 11.40 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 94.7 | false-accept absent 18.75 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 97.2 | false-accept absent 27.36 | |

AURC ĝ 22.41 vs ĝκ̂ 14.45; paired diff -7.96 [-8.90, -6.97]

**grec_testB** (n=16063, absent 29.1%, correct-if-answered-all 51.3%); default ĝ≥.5: coverage 71.6, selective risk 31.3, false-accept on absent 25.0

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 5.5 | 0.85 | 0.19 |
| joint[g] | 0.02 | 11.5 | 1.78 | 0.26 |
| joint[g] | 0.05 | 24.8 | 4.62 | 0.2 |
| joint[g] | 0.1 | 42.3 | 9.62 | 0.27 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 1.5 | 14.64 | True |
| selective_transfer[g] | 0.05 | 35.8 | 21.15 | True |
| selective_transfer[g] | 0.1 | 67.9 | 29.47 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 21.4 | 8.15 | 0.03 |
| joint[g*k] | 0.01 | 13.5 | 0.86 | 0.19 |
| joint[g*k] | 0.02 | 21.8 | 1.82 | 0.3 |
| joint[g*k] | 0.05 | 35.7 | 4.66 | 0.21 |
| joint[g*k] | 0.1 | 48.8 | 9.38 | 0.28 |
| selective_transfer[g*k] | 0.01 | 13.9 | 6.22 | True |
| selective_transfer[g*k] | 0.02 | 24.1 | 9.02 | True |
| selective_transfer[g*k] | 0.05 | 42.2 | 15.69 | True |
| selective_transfer[g*k] | 0.1 | 66.1 | 28.17 | True |
| absent_conformal[g] | 0.01 | present accepted 10.7 | false-accept absent 0.98 | |
| absent_conformal[g] | 0.02 | present accepted 21.6 | false-accept absent 2.04 | |
| absent_conformal[g] | 0.05 | present accepted 48.2 | false-accept absent 5.00 | |
| absent_conformal[g] | 0.1 | present accepted 72.8 | false-accept absent 9.89 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 75.5 | false-accept absent 10.85 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 83.3 | false-accept absent 14.32 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 90.2 | false-accept absent 23.33 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 93.0 | false-accept absent 31.86 | |

AURC ĝ 26.78 vs ĝκ̂ 22.11; paired diff -4.67 [-5.47, -3.97]

**ss2** (n=1272, absent 0.0%, correct-if-answered-all 91.4%); default ĝ≥.5: coverage 95.5, selective risk 7.2, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 13.8 | 0.94 | 0.02 |
| selective[g] | 0.1 | 95.0 | 7.20 | 0.03 |
| joint[g] | 0.01 | 20.6 | 0.49 | 0.1 |
| joint[g] | 0.02 | 43.1 | 1.28 | 0.12 |
| joint[g] | 0.05 | 78.9 | 3.90 | 0.13 |
| joint[g] | 0.1 | 99.0 | 8.20 | 0.03 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.1 | 74.7 | 4.53 | False |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.5 | 0.03 | 0.01 |
| selective[g*k] | 0.05 | 61.4 | 2.55 | 0.03 |
| selective[g*k] | 0.1 | 97.1 | 7.25 | 0.03 |
| joint[g*k] | 0.01 | 37.5 | 0.54 | 0.16 |
| joint[g*k] | 0.02 | 58.3 | 1.17 | 0.13 |
| joint[g*k] | 0.05 | 83.9 | 3.82 | 0.11 |
| joint[g*k] | 0.1 | 99.4 | 8.22 | 0.03 |
| selective_transfer[g*k] | 0.01 | 16.1 | 0.98 | False |
| selective_transfer[g*k] | 0.02 | 41.3 | 1.33 | False |
| selective_transfer[g*k] | 0.05 | 71.9 | 3.39 | False |
| selective_transfer[g*k] | 0.1 | 93.6 | 6.13 | False |

AURC ĝ 3.74 vs ĝκ̂ 2.35; paired diff -1.39 [-2.29, -0.55]

**venus** (n=6165, absent 14.6%, correct-if-answered-all 42.2%); default ĝ≥.5: coverage 69.1, selective risk 46.6, false-accept on absent 45.4

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 2.7 | 0.79 | 0.22 |
| joint[g] | 0.02 | 6.2 | 1.63 | 0.16 |
| joint[g] | 0.05 | 16.6 | 4.38 | 0.16 |
| joint[g] | 0.1 | 31.1 | 9.08 | 0.12 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.1 | 36.2 | 30.80 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 0.2 | 0.71 | 0.01 |
| joint[g*k] | 0.01 | 6.5 | 0.78 | 0.16 |
| joint[g*k] | 0.02 | 11.9 | 1.72 | 0.23 |
| joint[g*k] | 0.05 | 22.9 | 4.39 | 0.16 |
| joint[g*k] | 0.1 | 35.6 | 9.17 | 0.15 |
| selective_transfer[g*k] | 0.01 | 5.3 | 9.85 | True |
| selective_transfer[g*k] | 0.02 | 15.4 | 14.96 | True |
| selective_transfer[g*k] | 0.05 | 35.1 | 25.57 | True |
| selective_transfer[g*k] | 0.1 | 60.3 | 40.13 | True |
| absent_conformal[g] | 0.01 | present accepted 8.6 | false-accept absent 0.85 | |
| absent_conformal[g] | 0.02 | present accepted 13.0 | false-accept absent 1.98 | |
| absent_conformal[g] | 0.05 | present accepted 24.6 | false-accept absent 4.83 | |
| absent_conformal[g] | 0.1 | present accepted 39.8 | false-accept absent 9.72 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 37.3 | false-accept absent 8.45 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 50.3 | false-accept absent 15.46 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 70.4 | false-accept absent 42.16 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 81.1 | false-accept absent 62.07 | |

AURC ĝ 38.76 vs ĝκ̂ 33.52; paired diff -5.24 [-6.32, -4.30]

## Prop. 1 two-knob plug-in rule vs one-knob thresholds (expected cost per query, lower is better)

| set | costs | two-knob plug-in | ĝ≥.5 | ĝ thr tuned on calib | ĝκ̂ thr tuned | ĝ oracle thr | paired two−one(tuned) |
|---|---|---|---|---|---|---|---|
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0649 | 0.0640 | 0.0683 | 0.0664 | 0.0618 | -0.0033 [-0.0051, -0.0013] |
| refcoco_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1004 | 0.1224 | 0.1313 | 0.1026 | 0.1223 | -0.0309 [-0.0373, -0.0250] |
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.0718 | 0.0696 | 0.0732 | 0.0693 | 0.0626 | -0.0014 [-0.0023, -0.0005] |
| refcoco_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0690 | 0.0640 | 0.0726 | 0.0679 | 0.0618 | -0.0036 [-0.0056, -0.0015] |
| refcoco_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1395 | 0.2393 | 0.2327 | 0.1436 | 0.2327 | -0.0932 [-0.1091, -0.0762] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0918 | 0.0890 | 0.0948 | 0.0924 | 0.0859 | -0.0030 [-0.0052, -0.0009] |
| refcocog_test | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1366 | 0.1688 | 0.1783 | 0.1386 | 0.1684 | -0.0417 [-0.0483, -0.0351] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1023 | 0.0982 | 0.1048 | 0.0979 | 0.0874 | -0.0025 [-0.0038, -0.0014] |
| refcocog_test | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0981 | 0.0890 | 0.1025 | 0.0956 | 0.0859 | -0.0044 [-0.0065, -0.0021] |
| refcocog_test | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1899 | 0.3283 | 0.3008 | 0.1948 | 0.3008 | -0.1109 [-0.1283, -0.0937] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1133 | 0.1342 | 0.1212 | 0.1154 | 0.1146 | -0.0079 [-0.0101, -0.0057] |
| grec_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1397 | 0.2112 | 0.1806 | 0.1382 | 0.1799 | -0.0409 [-0.0473, -0.0355] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1377 | 0.1410 | 0.1359 | 0.1494 | 0.1338 | 0.0017 [0.0006, 0.0029] |
| grec_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1340 | 0.1846 | 0.1405 | 0.1439 | 0.1321 | -0.0064 [-0.0090, -0.0035] |
| grec_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1619 | 0.3652 | 0.3074 | 0.1603 | 0.3074 | -0.1455 [-0.1598, -0.1325] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1917 | 0.2118 | 0.2036 | 0.1941 | 0.2002 | -0.0119 [-0.0155, -0.0081] |
| grec_testA | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2268 | 0.3584 | 0.3212 | 0.2276 | 0.3187 | -0.0944 [-0.1044, -0.0827] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2320 | 0.2303 | 0.2299 | 0.2408 | 0.2292 | 0.0021 [0.0007, 0.0034] |
| grec_testA | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2187 | 0.2588 | 0.2297 | 0.2259 | 0.2228 | -0.0110 [-0.0152, -0.0070] |
| grec_testA | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2480 | 0.6514 | 0.5535 | 0.2487 | 0.5535 | -0.3055 [-0.3296, -0.2767] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2285 | 0.2571 | 0.2438 | 0.2290 | 0.2343 | -0.0152 [-0.0192, -0.0111] |
| grec_testB | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2595 | 0.4089 | 0.3612 | 0.2587 | 0.3584 | -0.1017 [-0.1139, -0.0884] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2897 | 0.2898 | 0.2869 | 0.3019 | 0.2849 | 0.0028 [0.0014, 0.0041] |
| grec_testB | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2686 | 0.3298 | 0.2829 | 0.2782 | 0.2703 | -0.0142 [-0.0182, -0.0100] |
| grec_testB | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2790 | 0.7124 | 0.6065 | 0.2773 | 0.6065 | -0.3275 [-0.3561, -0.2986] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1105 | 0.0916 | 0.1230 | 0.1093 | 0.0853 | -0.0126 [-0.0209, -0.0056] |
| ss2 | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1698 | 0.1608 | 0.1930 | 0.1859 | 0.1561 | -0.0232 [-0.0375, -0.0089] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1195 | 0.1140 | 0.1313 | 0.1038 | 0.0881 | -0.0118 [-0.0193, -0.0047] |
| ss2 | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1266 | 0.0916 | 0.1466 | 0.1191 | 0.0853 | -0.0200 [-0.0277, -0.0126] |
| ss2 | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2453 | 0.2991 | 0.2622 | 0.2547 | 0.2496 | -0.0169 [-0.0486, 0.0137] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.3635 | 0.4369 | 0.3706 | 0.3668 | 0.3639 | -0.0071 [-0.0129, -0.0015] |
| venus | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.4401 | 0.6932 | 0.4633 | 0.4218 | 0.4633 | -0.0232 [-0.0358, -0.0114] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.5525 | 0.5513 | 0.5534 | 0.5534 | 0.5512 | -0.0010 [-0.0047, 0.0028] |
| venus | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.3827 | 0.5031 | 0.3843 | 0.3955 | 0.3796 | -0.0015 [-0.0084, 0.0056] |
| venus | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.5232 | 1.2058 | 0.6622 | 0.5101 | 0.6622 | -0.1390 [-0.1675, -0.1127] |
