# sprint_abs tables (v1.2)

## GREC official protocol (gRefCOCO; Pr@(F1=1, GIoU>=0.5), N-acc, T-acc)

| split | variant | Pr@F1=1 | 95% CI | N-acc | T-acc | Pr single | Pr multi | Pr (plain IoU) |
|---|---|---|---|---|---|---|---|---|
| grec_val | single_raw@0.7(official) | 70.8 | 70.8 [70.3, 71.2] | 87.8 | 94.8 | 91.5 | 0.0 | 70.9 |
| grec_val | single_raw@0.5 | 69.6 | 69.6 [69.1, 70.1] | 83.8 | 97.0 | 92.2 | 0.0 | 69.8 |
| grec_val | single_platt@0.7 | 70.5 | 70.5 [70.0, 71.0] | 86.9 | 95.4 | 91.7 | 0.0 | 70.7 |
| grec_val | multi_platt@0.7(official) | 70.5 | 70.5 [70.0, 71.0] | 86.9 | 95.4 | 91.7 | 0.0 | 70.7 |
| grec_val | multi_raw@0.7(official) | 70.8 | 70.8 [70.3, 71.2] | 87.8 | 94.8 | 91.5 | 0.0 | 70.9 |
| grec_val | multi_platt@tuned | 70.2 | 70.2 [69.7, 70.7] | 84.7 | 96.5 | 91.8 | 2.1 | 70.4 |
| grec_val | single_platt@tuned | 69.9 | 69.9 [69.4, 70.4] | 84.7 | 96.5 | 92.1 | 0.0 | 70.1 |
| grec_val | paired diff multi-single (official) | -0.2 [-0.3, -0.1] | | | | | | |
| grec_testA | single_raw@0.7(official) | 46.5 | 46.5 [45.8, 47.2] | 79.4 | 90.3 | 91.2 | 0.0 | 46.6 |
| grec_testA | single_raw@0.5 | 45.5 | 45.5 [44.8, 46.2] | 74.3 | 93.3 | 91.9 | 0.0 | 45.6 |
| grec_testA | single_platt@0.7 | 46.3 | 46.3 [45.6, 47.0] | 78.1 | 91.2 | 91.5 | 0.0 | 46.3 |
| grec_testA | multi_platt@0.7(official) | 46.3 | 46.3 [45.6, 47.0] | 78.1 | 91.2 | 91.5 | 0.0 | 46.3 |
| grec_testA | multi_raw@0.7(official) | 46.5 | 46.5 [45.8, 47.2] | 79.4 | 90.3 | 91.2 | 0.0 | 46.6 |
| grec_testA | multi_platt@tuned | 46.4 | 46.4 [45.7, 47.1] | 75.4 | 92.8 | 91.6 | 1.6 | 46.5 |
| grec_testA | single_platt@tuned | 45.7 | 45.7 [45.0, 46.4] | 75.4 | 92.8 | 91.7 | 0.0 | 45.8 |
| grec_testA | paired diff multi-single (official) | -0.2 [-0.3, -0.1] | | | | | | |
| grec_testB | single_raw@0.7(official) | 50.5 | 50.5 [49.5, 51.6] | 76.5 | 87.9 | 80.4 | 0.0 | 50.7 |
| grec_testB | single_raw@0.5 | 49.3 | 49.3 [48.3, 50.4] | 71.6 | 90.6 | 81.0 | 0.0 | 49.5 |
| grec_testB | single_platt@0.7 | 50.2 | 50.2 [49.2, 51.3] | 75.3 | 88.6 | 80.5 | 0.0 | 50.4 |
| grec_testB | multi_platt@0.7(official) | 50.2 | 50.2 [49.2, 51.3] | 75.3 | 88.6 | 80.5 | 0.0 | 50.4 |
| grec_testB | multi_raw@0.7(official) | 50.5 | 50.5 [49.5, 51.6] | 76.5 | 87.9 | 80.4 | 0.0 | 50.7 |
| grec_testB | multi_platt@tuned | 49.9 | 49.9 [48.9, 50.9] | 72.9 | 90.1 | 80.7 | 0.8 | 50.0 |
| grec_testB | single_platt@tuned | 49.6 | 49.6 [48.6, 50.7] | 72.9 | 90.1 | 80.9 | 0.0 | 49.8 |
| grec_testB | paired diff multi-single (official) | -0.3 [-0.4, -0.2] | | | | | | |
| grefcalib | single_raw@0.7(official) | 62.8 | - | 95.8 | 94.6 | 90.8 | 0.0 | 63.1 |
| grefcalib | single_raw@0.5 | 63.0 | - | 94.5 | 96.3 | 91.4 | 0.0 | 63.3 |
| grefcalib | single_platt@0.7 | 62.8 | - | 95.3 | 94.9 | 90.9 | 0.0 | 63.1 |
| grefcalib | multi_platt@0.7(official) | 62.8 | - | 95.3 | 94.9 | 90.9 | 0.0 | 63.1 |
| grefcalib | multi_raw@0.7(official) | 62.8 | - | 95.8 | 94.6 | 90.8 | 0.0 | 63.1 |
| grefcalib | multi_platt@tuned | 63.7 | - | 95.1 | 96.0 | 90.9 | 2.8 | 64.0 |
| grefcalib | single_platt@tuned | 63.0 | - | 95.1 | 96.0 | 91.4 | 0.0 | 63.4 |

Tuned on grefcalib: {"pr_f1_grefcalib": 0.6365300984206912, "t_presence": 0.5999999999999999, "t_extra": 0.35, "single_t_presence": 0.5999999999999999, "single_pr_f1_grefcalib": 0.6303501945525292}; extra-box model: {"n": 849, "pos": 0.19434628975265017}

## VenusBench-GD, official UI-Venus evaluator

| config | Element | Visual | Spatial | Basic | Reasoning | Functional | Refusal | Advanced | Overall | n |
|---|---|---|---|---|---|---|---|---|---|---|
| platt_gated | 54.62 | 67.34 | 51.21 | 56.69 | 11.92 | 40.71 | 70.41 | 38.80 | 48.84 | 6165 |
| raw_gated | 48.02 | 59.12 | 45.00 | 49.81 | 9.21 | 37.43 | 83.43 | 41.17 | 46.02 | 6165 |
| raw_ungated | 65.93 | 83.91 | 68.12 | 70.95 | 16.44 | 45.57 | 83.43 | 46.23 | 60.10 | 6165 |
| stack_gated | 58.77 | 80.69 | 62.20 | 65.11 | 4.70 | 39.29 | 45.05 | 27.05 | 48.40 | 6165 |

## Presence ĝ calibration (fit on held-out training-distribution splits)

| set | subset | method | AUROC | N-acc@.5 | recall@.5 | ECE15 | ECE15-mass | ECE10 | Brier | NLL | +EM prior: ECE15 / N-acc / π̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grec_val | all | raw | 96.9 | 83.8 | 97.0 | 3.3 | 3.2 | 3.2 | 6.03 | 0.219 | - |
| grec_val | all | temp | 96.9 | 83.8 | 97.0 | 3.3 | 3.2 | 3.2 | 6.03 | 0.219 | 2.4 / 85.1 / 0.662 |
| grec_val | all | platt | 96.9 | 82.7 | 97.3 | 3.8 | 3.8 | 3.8 | 6.15 | 0.223 | 3.0 / 84.0 / 0.670 |
| grec_val | all | beta | 96.9 | 83.2 | 97.2 | 3.6 | 3.7 | 3.7 | 6.11 | 0.222 | 2.8 / 84.5 / 0.669 |
| grec_val | all | iso | 96.9 | 82.2 | 97.4 | 3.7 | 3.6 | 3.7 | 6.16 | 0.223 | 3.0 / 84.3 / 0.670 |
| grec_val | all | multi | 97.0 | 81.9 | 97.6 | 3.6 | 3.7 | 3.6 | 6.09 | 0.214 | 2.5 / 83.4 / 0.669 |
| grec_val | gref_only | raw | 94.0 | 83.8 | 93.2 | 7.4 | 7.3 | 7.3 | 9.94 | 0.357 | - |
| grec_val | gref_only | temp | 94.0 | 83.8 | 93.2 | 7.4 | 7.3 | 7.3 | 9.95 | 0.357 | 4.6 / 90.7 / 0.348 |
| grec_val | gref_only | platt | 94.0 | 82.7 | 93.9 | 8.3 | 8.3 | 8.2 | 10.25 | 0.368 | 3.8 / 89.9 / 0.362 |
| grec_val | gref_only | beta | 94.0 | 83.2 | 93.6 | 8.0 | 8.1 | 7.9 | 10.15 | 0.367 | 4.1 / 90.2 / 0.361 |
| grec_val | gref_only | iso | 94.0 | 82.2 | 94.2 | 7.9 | 8.0 | 7.9 | 10.25 | 0.368 | 3.8 / 90.2 / 0.358 |
| grec_val | gref_only | multi | 94.3 | 81.9 | 94.7 | 8.5 | 8.7 | 8.5 | 10.16 | 0.349 | 3.6 / 90.8 / 0.350 |
| grec_testA | all | raw | 91.6 | 74.3 | 93.3 | 5.6 | 5.6 | 5.6 | 8.86 | 0.368 | - |
| grec_testA | all | temp | 91.6 | 74.3 | 93.3 | 5.6 | 5.6 | 5.6 | 8.86 | 0.369 | 5.7 / 73.0 / 0.770 |
| grec_testA | all | platt | 91.6 | 73.1 | 93.8 | 5.5 | 5.3 | 5.4 | 8.76 | 0.362 | 5.6 / 71.0 / 0.780 |
| grec_testA | all | beta | 91.6 | 73.5 | 93.6 | 5.5 | 5.4 | 5.5 | 8.77 | 0.357 | 5.5 / 71.7 / 0.779 |
| grec_testA | all | iso | 91.6 | 72.9 | 94.1 | 5.6 | 5.1 | 5.6 | 8.81 | 0.352 | 5.7 / 70.4 / 0.779 |
| grec_testA | all | multi | 92.1 | 72.7 | 94.5 | 4.2 | 3.9 | 4.2 | 8.27 | 0.325 | 4.5 / 70.4 / 0.783 |
| grec_testA | gref_only | raw | 89.3 | 74.3 | 91.2 | 7.1 | 7.2 | 7.0 | 11.39 | 0.425 | - |
| grec_testA | gref_only | temp | 89.3 | 74.3 | 91.2 | 7.1 | 7.2 | 7.1 | 11.39 | 0.426 | 7.2 / 76.3 / 0.653 |
| grec_testA | gref_only | platt | 89.3 | 73.1 | 91.9 | 7.0 | 7.0 | 6.9 | 11.31 | 0.422 | 6.9 / 74.6 / 0.668 |
| grec_testA | gref_only | beta | 89.3 | 73.5 | 91.6 | 6.9 | 7.0 | 6.9 | 11.30 | 0.422 | 6.9 / 75.1 / 0.665 |
| grec_testA | gref_only | iso | 89.3 | 72.9 | 92.3 | 7.1 | 6.9 | 7.1 | 11.36 | 0.424 | 6.7 / 75.0 / 0.666 |
| grec_testA | gref_only | multi | 90.2 | 72.7 | 92.9 | 5.4 | 5.0 | 5.4 | 10.65 | 0.382 | 4.9 / 74.7 / 0.667 |
| grec_testB | all | raw | 87.2 | 71.6 | 90.6 | 9.0 | 9.0 | 9.0 | 12.64 | 0.671 | - |
| grec_testB | all | temp | 87.2 | 71.6 | 90.6 | 9.1 | 9.0 | 9.0 | 12.64 | 0.673 | 8.9 / 72.1 / 0.705 |
| grec_testB | all | platt | 87.2 | 70.1 | 91.1 | 9.2 | 9.1 | 9.2 | 12.69 | 0.660 | 9.1 / 70.3 / 0.716 |
| grec_testB | all | beta | 87.2 | 70.6 | 91.0 | 9.0 | 9.0 | 9.0 | 12.65 | 0.635 | 9.0 / 71.0 / 0.714 |
| grec_testB | all | iso | 87.2 | 69.6 | 91.3 | 9.4 | 9.2 | 9.4 | 12.70 | 0.605 | 9.3 / 70.9 / 0.714 |
| grec_testB | all | multi | 87.3 | 69.5 | 91.2 | 8.3 | 8.2 | 8.3 | 12.49 | 0.589 | 8.2 / 69.9 / 0.713 |
| grec_testB | gref_only | raw | 87.4 | 71.6 | 92.1 | 9.3 | 9.3 | 9.2 | 13.84 | 0.527 | - |
| grec_testB | gref_only | temp | 87.4 | 71.6 | 92.1 | 9.3 | 9.3 | 9.3 | 13.84 | 0.528 | 8.1 / 75.3 / 0.574 |
| grec_testB | gref_only | platt | 87.4 | 70.1 | 92.9 | 9.8 | 9.6 | 9.8 | 13.99 | 0.534 | 8.1 / 73.7 / 0.591 |
| grec_testB | gref_only | beta | 87.4 | 70.6 | 92.6 | 9.5 | 9.4 | 9.5 | 13.93 | 0.534 | 8.1 / 74.5 / 0.588 |
| grec_testB | gref_only | iso | 87.4 | 69.6 | 93.2 | 9.9 | 9.8 | 9.9 | 14.01 | 0.536 | 8.0 / 73.1 / 0.587 |
| grec_testB | gref_only | multi | 87.8 | 69.5 | 93.1 | 9.1 | 8.8 | 8.9 | 13.70 | 0.498 | 6.5 / 74.3 / 0.584 |
| calib_natural | all | raw | 99.2 | 93.8 | 98.2 | 0.6 | 0.5 | 0.6 | 2.37 | 0.090 | - |
| calib_natural | all | temp | 99.2 | 93.8 | 98.2 | 0.6 | 0.5 | 0.5 | 2.37 | 0.090 | 0.6 / 94.0 / 0.718 |
| calib_natural | all | platt | 99.2 | 93.5 | 98.4 | 0.4 | 0.3 | 0.2 | 2.36 | 0.090 | 0.4 / 93.5 / 0.723 |
| calib_natural | all | beta | 99.2 | 93.6 | 98.3 | 0.4 | 0.3 | 0.3 | 2.36 | 0.089 | 0.4 / 93.6 / 0.723 |
| calib_natural | all | iso | 99.2 | 93.4 | 98.5 | 0.2 | 0.1 | 0.2 | 2.34 | 0.089 | 0.2 / 93.4 / 0.723 |
| calib_natural | all | multi | 99.2 | 93.3 | 98.5 | 1.1 | 0.9 | 1.1 | 2.29 | 0.088 | 1.1 / 93.3 / 0.721 |
| grefcalib | all | raw | 98.5 | 94.5 | 96.3 | 4.6 | 4.6 | 4.6 | 3.05 | 0.130 | - |
| grefcalib | all | temp | 98.5 | 94.5 | 96.3 | 4.6 | 4.6 | 4.6 | 3.05 | 0.130 | 2.2 / 92.7 / 0.874 |
| grefcalib | all | platt | 98.5 | 94.0 | 96.6 | 4.1 | 4.1 | 4.1 | 2.83 | 0.120 | 1.8 / 91.6 / 0.879 |
| grefcalib | all | beta | 98.5 | 94.5 | 96.5 | 4.1 | 4.2 | 4.1 | 2.88 | 0.118 | 1.8 / 92.2 / 0.879 |
| grefcalib | all | iso | 98.5 | 94.0 | 96.8 | 4.1 | 4.2 | 4.1 | 2.86 | 0.116 | 1.9 / 93.6 / 0.878 |
| grefcalib | all | multi | 98.6 | 94.9 | 97.2 | 4.3 | 4.3 | 4.3 | 2.47 | 0.106 | 1.6 / 90.2 / 0.882 |
| guisyn_test | natural_fit | raw | 86.3 | 69.8 | 90.3 | 8.7 | 8.7 | 8.7 | 11.80 | 0.492 | - |
| guisyn_test | natural_fit | temp | 86.3 | 69.8 | 90.3 | 8.7 | 8.7 | 8.7 | 11.80 | 0.493 | 8.4 / 68.7 / 0.752 |
| guisyn_test | natural_fit | platt | 86.3 | 68.2 | 91.2 | 8.1 | 8.0 | 8.1 | 11.56 | 0.484 | 7.7 / 66.6 / 0.765 |
| guisyn_test | natural_fit | beta | 86.3 | 68.9 | 90.8 | 8.2 | 8.2 | 8.2 | 11.59 | 0.487 | 7.8 / 67.2 / 0.763 |
| guisyn_test | natural_fit | iso | 86.0 | 67.6 | 91.5 | 8.1 | 8.0 | 8.0 | 11.67 | 0.509 | 8.0 / 65.6 / 0.763 |
| guisyn_test | natural_fit | multi | 81.7 | 71.5 | 80.9 | 15.4 | 15.4 | 15.4 | 16.62 | 0.602 | 18.6 / 74.8 / 0.622 |
| guisyn_test | gui_fit | raw | 86.3 | 69.8 | 90.3 | 8.7 | 8.7 | 8.7 | 11.80 | 0.492 | - |
| guisyn_test | gui_fit | temp | 86.3 | 69.8 | 90.3 | 6.9 | 7.3 | 7.0 | 11.30 | 0.373 | 15.2 / 79.4 / 0.606 |
| guisyn_test | gui_fit | platt | 86.3 | 58.8 | 94.9 | 3.9 | 4.2 | 3.7 | 10.72 | 0.360 | 4.0 / 62.1 / 0.744 |
| guisyn_test | gui_fit | beta | 86.3 | 55.9 | 95.6 | 2.7 | 3.4 | 2.6 | 10.66 | 0.355 | 2.6 / 58.3 / 0.744 |
| guisyn_test | gui_fit | iso | 86.3 | 60.7 | 94.4 | 0.7 | 1.3 | 0.6 | 10.52 | 0.347 | 1.3 / 61.6 / 0.742 |
| guisyn_test | gui_fit | multi | 96.5 | 84.1 | 95.6 | 0.8 | 0.8 | 0.8 | 5.64 | 0.198 | 0.7 / 84.4 / 0.760 |
| venus | natural_fit | raw | 79.3 | 83.4 | 60.2 | 32.8 | 32.9 | 32.7 | 29.13 | 1.154 | - |
| venus | natural_fit | temp | 79.3 | 83.4 | 60.2 | 32.7 | 32.9 | 32.7 | 29.14 | 1.156 | 45.8 / 91.1 / 0.396 |
| venus | natural_fit | platt | 79.3 | 82.1 | 62.3 | 30.9 | 31.0 | 30.8 | 27.64 | 1.069 | 43.0 / 90.2 / 0.424 |
| venus | natural_fit | beta | 79.3 | 82.8 | 61.2 | 31.4 | 31.4 | 31.2 | 27.80 | 1.033 | 43.7 / 90.4 / 0.417 |
| venus | natural_fit | iso | 79.3 | 81.5 | 63.1 | 31.2 | 31.3 | 31.1 | 28.07 | 1.019 | 43.4 / 90.7 / 0.420 |
| venus | natural_fit | multi | 80.2 | 90.8 | 47.4 | 42.8 | 42.8 | 42.8 | 35.15 | 1.277 | 68.2 / 98.6 / 0.172 |
| venus | gui_fit | raw | 79.3 | 83.4 | 60.2 | 32.8 | 32.9 | 32.7 | 29.13 | 1.154 | - |
| venus | gui_fit | temp | 79.3 | 83.4 | 60.2 | 34.1 | 34.1 | 34.1 | 25.00 | 0.742 | 85.3 / 100.0 / 0.001 |
| venus | gui_fit | platt | 79.3 | 70.4 | 72.7 | 24.9 | 24.9 | 24.9 | 18.61 | 0.556 | 72.2 / 99.9 / 0.133 |
| venus | gui_fit | beta | 79.3 | 67.0 | 75.1 | 24.0 | 24.0 | 24.0 | 18.78 | 0.591 | 68.9 / 99.9 / 0.165 |
| venus | gui_fit | iso | 79.3 | 72.5 | 71.2 | 24.5 | 24.5 | 24.5 | 19.29 | 0.588 | 64.0 / 100.0 / 0.214 |
| venus | gui_fit | multi | 60.5 | 45.1 | 67.3 | 26.1 | 26.1 | 26.1 | 27.70 | 0.893 | 34.6 / 61.4 / 0.511 |

## κ̂ calibration (natural: event IoU>=0.5; GUI: click in box)

| set | fit | method | AUROC | AURC | ECE15 | ECE15-mass | Brier | mean κ̂ | rate |
|---|---|---|---|---|---|---|---|---|---|
| refcoco_val | natural_fit | raw | 90.9 | 0.96 | 0.7 | 0.5 | 4.29 | 93.9 | 93.5 |
| refcoco_val | natural_fit | temp | 90.9 | 0.96 | 0.4 | 0.3 | 4.29 | 93.5 | 93.5 |
| refcoco_val | natural_fit | platt | 90.9 | 0.96 | 0.6 | 0.6 | 4.30 | 93.2 | 93.5 |
| refcoco_val | natural_fit | beta | 90.9 | 0.96 | 0.8 | 0.7 | 4.31 | 93.2 | 93.5 |
| refcoco_val | natural_fit | iso | 90.9 | 0.96 | 0.9 | 0.7 | 4.30 | 93.1 | 93.5 |
| refcoco_val | natural_fit | multi | 91.0 | 0.98 | 1.0 | 0.9 | 4.27 | 92.6 | 93.5 |
| refcoco_testA | natural_fit | raw | 92.4 | 0.60 | 0.9 | 0.9 | 3.27 | 95.7 | 95.1 |
| refcoco_testA | natural_fit | temp | 92.4 | 0.60 | 0.8 | 0.7 | 3.26 | 95.4 | 95.1 |
| refcoco_testA | natural_fit | platt | 92.4 | 0.60 | 0.6 | 0.5 | 3.25 | 95.2 | 95.1 |
| refcoco_testA | natural_fit | beta | 92.4 | 0.60 | 0.6 | 0.5 | 3.26 | 95.2 | 95.1 |
| refcoco_testA | natural_fit | iso | 92.4 | 0.60 | 0.7 | 0.4 | 3.27 | 95.2 | 95.1 |
| refcoco_testA | natural_fit | multi | 92.3 | 0.62 | 0.7 | 0.5 | 3.26 | 94.6 | 95.1 |
| refcoco_testB | natural_fit | raw | 91.1 | 1.24 | 0.8 | 0.4 | 5.36 | 91.3 | 91.5 |
| refcoco_testB | natural_fit | temp | 91.1 | 1.24 | 1.2 | 0.8 | 5.37 | 90.8 | 91.5 |
| refcoco_testB | natural_fit | platt | 91.1 | 1.24 | 1.4 | 1.2 | 5.39 | 90.3 | 91.5 |
| refcoco_testB | natural_fit | beta | 91.1 | 1.24 | 1.5 | 1.3 | 5.42 | 90.2 | 91.5 |
| refcoco_testB | natural_fit | iso | 91.1 | 1.24 | 1.8 | 1.5 | 5.42 | 90.1 | 91.5 |
| refcoco_testB | natural_fit | multi | 91.2 | 1.24 | 2.0 | 1.8 | 5.41 | 89.7 | 91.5 |
| refcocoplus_val | natural_fit | raw | 90.3 | 1.70 | 1.2 | 1.2 | 6.20 | 91.0 | 90.0 |
| refcocoplus_val | natural_fit | temp | 90.3 | 1.70 | 0.8 | 0.9 | 6.19 | 90.5 | 90.0 |
| refcocoplus_val | natural_fit | platt | 90.3 | 1.70 | 1.1 | 1.2 | 6.20 | 90.0 | 90.0 |
| refcocoplus_val | natural_fit | beta | 90.3 | 1.70 | 1.2 | 1.1 | 6.21 | 90.0 | 90.0 |
| refcocoplus_val | natural_fit | iso | 90.3 | 1.70 | 1.1 | 0.9 | 6.20 | 89.9 | 90.0 |
| refcocoplus_val | natural_fit | multi | 90.3 | 1.72 | 0.6 | 0.8 | 6.22 | 89.6 | 90.0 |
| refcocoplus_testA | natural_fit | raw | 92.2 | 0.86 | 1.2 | 1.0 | 4.42 | 93.9 | 93.2 |
| refcocoplus_testA | natural_fit | temp | 92.2 | 0.86 | 1.0 | 0.9 | 4.41 | 93.5 | 93.2 |
| refcocoplus_testA | natural_fit | platt | 92.2 | 0.86 | 0.9 | 0.9 | 4.41 | 93.2 | 93.2 |
| refcocoplus_testA | natural_fit | beta | 92.2 | 0.86 | 1.1 | 0.9 | 4.43 | 93.2 | 93.2 |
| refcocoplus_testA | natural_fit | iso | 92.2 | 0.86 | 1.1 | 0.9 | 4.41 | 93.1 | 93.2 |
| refcocoplus_testA | natural_fit | multi | 92.4 | 0.84 | 1.6 | 1.3 | 4.36 | 92.6 | 93.2 |
| refcocoplus_testB | natural_fit | raw | 87.1 | 3.29 | 1.9 | 1.8 | 8.50 | 87.3 | 86.0 |
| refcocoplus_testB | natural_fit | temp | 87.1 | 3.29 | 1.5 | 1.6 | 8.49 | 86.8 | 86.0 |
| refcocoplus_testB | natural_fit | platt | 87.1 | 3.29 | 1.9 | 1.7 | 8.51 | 86.0 | 86.0 |
| refcocoplus_testB | natural_fit | beta | 87.1 | 3.29 | 2.2 | 2.2 | 8.55 | 86.0 | 86.0 |
| refcocoplus_testB | natural_fit | iso | 87.1 | 3.29 | 1.9 | 2.1 | 8.52 | 85.8 | 86.0 |
| refcocoplus_testB | natural_fit | multi | 86.8 | 3.37 | 1.8 | 1.6 | 8.61 | 85.9 | 86.0 |
| refcocog_val | natural_fit | raw | 87.1 | 1.88 | 2.0 | 1.3 | 6.24 | 92.2 | 91.0 |
| refcocog_val | natural_fit | temp | 87.1 | 1.88 | 1.6 | 0.9 | 6.23 | 91.7 | 91.0 |
| refcocog_val | natural_fit | platt | 87.1 | 1.88 | 1.6 | 1.3 | 6.25 | 91.3 | 91.0 |
| refcocog_val | natural_fit | beta | 87.1 | 1.88 | 1.8 | 1.5 | 6.27 | 91.2 | 91.0 |
| refcocog_val | natural_fit | iso | 87.1 | 1.88 | 1.8 | 1.4 | 6.26 | 91.1 | 91.0 |
| refcocog_val | natural_fit | multi | 86.7 | 1.96 | 1.4 | 1.3 | 6.26 | 91.3 | 91.0 |
| refcocog_test | natural_fit | raw | 88.5 | 1.70 | 1.5 | 1.2 | 6.10 | 92.2 | 91.0 |
| refcocog_test | natural_fit | temp | 88.5 | 1.70 | 1.1 | 0.8 | 6.08 | 91.7 | 91.0 |
| refcocog_test | natural_fit | platt | 88.5 | 1.70 | 0.9 | 0.8 | 6.08 | 91.3 | 91.0 |
| refcocog_test | natural_fit | beta | 88.5 | 1.70 | 1.1 | 0.9 | 6.08 | 91.2 | 91.0 |
| refcocog_test | natural_fit | iso | 88.5 | 1.70 | 1.1 | 0.9 | 6.10 | 91.1 | 91.0 |
| refcocog_test | natural_fit | multi | 88.0 | 1.82 | 1.2 | 0.7 | 6.08 | 91.2 | 91.0 |
| grec_val | natural_fit | raw | 91.1 | 0.94 | 0.6 | 0.5 | 4.27 | 93.9 | 93.5 |
| grec_val | natural_fit | temp | 91.1 | 0.94 | 0.4 | 0.4 | 4.26 | 93.5 | 93.5 |
| grec_val | natural_fit | platt | 91.1 | 0.94 | 0.6 | 0.7 | 4.27 | 93.2 | 93.5 |
| grec_val | natural_fit | beta | 91.1 | 0.94 | 0.8 | 0.7 | 4.28 | 93.2 | 93.5 |
| grec_val | natural_fit | iso | 91.1 | 0.94 | 0.8 | 0.8 | 4.28 | 93.1 | 93.5 |
| grec_val | natural_fit | multi | 91.2 | 0.96 | 1.0 | 0.9 | 4.25 | 92.6 | 93.5 |
| grec_testA | natural_fit | raw | 94.2 | 0.76 | 1.5 | 1.5 | 3.63 | 94.3 | 93.0 |
| grec_testA | natural_fit | temp | 94.2 | 0.76 | 1.3 | 1.4 | 3.63 | 94.0 | 93.0 |
| grec_testA | natural_fit | platt | 94.2 | 0.76 | 1.0 | 1.0 | 3.59 | 93.8 | 93.0 |
| grec_testA | natural_fit | beta | 94.2 | 0.76 | 1.2 | 1.1 | 3.63 | 93.9 | 93.0 |
| grec_testA | natural_fit | iso | 94.2 | 0.76 | 1.1 | 0.8 | 3.59 | 93.7 | 93.0 |
| grec_testA | natural_fit | multi | 94.0 | 0.79 | 1.3 | 1.1 | 3.64 | 93.3 | 93.0 |
| grec_testB | natural_fit | raw | 94.2 | 2.52 | 2.5 | 2.4 | 6.61 | 85.7 | 83.6 |
| grec_testB | natural_fit | temp | 94.2 | 2.52 | 2.6 | 2.4 | 6.63 | 85.3 | 83.6 |
| grec_testB | natural_fit | platt | 94.2 | 2.52 | 2.0 | 1.7 | 6.53 | 84.6 | 83.6 |
| grec_testB | natural_fit | beta | 94.2 | 2.52 | 2.5 | 2.0 | 6.67 | 84.9 | 83.6 |
| grec_testB | natural_fit | iso | 94.2 | 2.52 | 2.1 | 1.7 | 6.60 | 84.4 | 83.6 |
| grec_testB | natural_fit | multi | 94.2 | 2.52 | 2.6 | 2.1 | 6.61 | 84.2 | 83.6 |
| calib_natural | natural_fit | raw | 92.0 | 1.18 | 1.2 | 1.1 | 5.31 | 92.3 | 91.5 |
| calib_natural | natural_fit | temp | 92.0 | 1.18 | 1.0 | 0.9 | 5.30 | 91.9 | 91.5 |
| calib_natural | natural_fit | platt | 92.0 | 1.18 | 0.7 | 0.7 | 5.29 | 91.5 | 91.5 |
| calib_natural | natural_fit | beta | 92.0 | 1.18 | 0.6 | 0.5 | 5.29 | 91.5 | 91.5 |
| calib_natural | natural_fit | iso | 92.0 | 1.18 | 0.6 | 0.4 | 5.27 | 91.4 | 91.5 |
| calib_natural | natural_fit | multi | 92.2 | 1.17 | 0.8 | 0.7 | 5.19 | 91.1 | 91.5 |
| guisyn_test | natural_fit | raw | 89.7 | 2.52 | 18.0 | 18.0 | 11.69 | 69.3 | 87.3 |
| guisyn_test | natural_fit | temp | 89.7 | 2.52 | 18.6 | 18.6 | 11.78 | 68.7 | 87.3 |
| guisyn_test | natural_fit | platt | 89.7 | 2.52 | 20.3 | 20.3 | 12.74 | 67.0 | 87.3 |
| guisyn_test | natural_fit | beta | 89.7 | 2.52 | 19.9 | 19.9 | 12.40 | 67.4 | 87.3 |
| guisyn_test | natural_fit | iso | 89.6 | 2.56 | 21.3 | 21.3 | 13.00 | 66.0 | 87.3 |
| guisyn_test | natural_fit | multi | 80.5 | 4.22 | 73.4 | 73.4 | 65.47 | 13.9 | 87.3 |
| guisyn_test | gui_fit | raw | 89.7 | 2.52 | 18.0 | 18.0 | 11.69 | 69.3 | 87.3 |
| guisyn_test | gui_fit | temp | 89.7 | 2.52 | 15.8 | 15.8 | 11.55 | 71.5 | 87.3 |
| guisyn_test | gui_fit | platt | 89.7 | 2.52 | 0.8 | 0.8 | 7.21 | 87.3 | 87.3 |
| guisyn_test | gui_fit | beta | 89.7 | 2.52 | 0.4 | 0.5 | 7.20 | 87.2 | 87.3 |
| guisyn_test | gui_fit | iso | 89.7 | 2.52 | 0.5 | 0.6 | 7.20 | 87.2 | 87.3 |
| guisyn_test | gui_fit | multi | 89.6 | 2.55 | 0.9 | 0.8 | 7.14 | 86.9 | 87.3 |
| ss2 | natural_fit | raw | 78.2 | 3.35 | 30.0 | 30.0 | 20.34 | 59.9 | 89.9 |
| ss2 | natural_fit | temp | 78.2 | 3.35 | 30.3 | 30.3 | 20.19 | 59.6 | 89.9 |
| ss2 | natural_fit | platt | 78.2 | 3.35 | 32.6 | 32.6 | 22.10 | 57.3 | 89.9 |
| ss2 | natural_fit | beta | 78.2 | 3.35 | 31.7 | 31.7 | 20.73 | 58.2 | 89.9 |
| ss2 | natural_fit | iso | 78.1 | 3.37 | 32.7 | 32.7 | 21.85 | 57.1 | 89.9 |
| ss2 | natural_fit | multi | 71.9 | 4.86 | 78.5 | 78.5 | 73.15 | 11.3 | 89.9 |
| ss2 | gui_fit | raw | 78.2 | 3.35 | 30.0 | 30.0 | 20.34 | 59.9 | 89.9 |
| ss2 | gui_fit | temp | 78.2 | 3.35 | 28.9 | 28.9 | 21.18 | 61.0 | 89.9 |
| ss2 | gui_fit | platt | 78.2 | 3.35 | 6.2 | 7.2 | 9.32 | 83.7 | 89.9 |
| ss2 | gui_fit | beta | 78.2 | 3.35 | 6.7 | 7.6 | 9.39 | 83.3 | 89.9 |
| ss2 | gui_fit | iso | 78.2 | 3.35 | 6.7 | 7.5 | 9.40 | 83.4 | 89.9 |
| ss2 | gui_fit | multi | 78.9 | 3.26 | 6.4 | 7.0 | 8.99 | 83.5 | 89.9 |
| venus | natural_fit | raw | 82.4 | 21.71 | 17.8 | 17.7 | 20.86 | 38.4 | 56.1 |
| venus | natural_fit | temp | 82.4 | 21.71 | 17.5 | 17.5 | 20.67 | 38.6 | 56.1 |
| venus | natural_fit | platt | 82.4 | 21.71 | 20.0 | 20.0 | 21.83 | 36.1 | 56.1 |
| venus | natural_fit | beta | 82.4 | 21.71 | 17.0 | 17.3 | 20.71 | 39.2 | 56.1 |
| venus | natural_fit | iso | 81.3 | 22.70 | 21.3 | 21.3 | 22.28 | 34.8 | 56.1 |
| venus | natural_fit | multi | 76.6 | 24.74 | 54.0 | 54.0 | 52.88 | 2.1 | 56.1 |
| venus | gui_fit | raw | 82.4 | 21.71 | 17.8 | 17.7 | 20.86 | 38.4 | 56.1 |
| venus | gui_fit | temp | 82.4 | 21.71 | 19.0 | 19.0 | 21.78 | 37.5 | 56.1 |
| venus | gui_fit | platt | 82.4 | 21.71 | 6.9 | 6.5 | 17.34 | 62.6 | 56.1 |
| venus | gui_fit | beta | 82.4 | 21.71 | 7.1 | 6.8 | 17.30 | 62.9 | 56.1 |
| venus | gui_fit | iso | 82.4 | 21.71 | 7.2 | 6.7 | 17.35 | 62.8 | 56.1 |
| venus | gui_fit | multi | 82.8 | 21.47 | 5.8 | 5.5 | 16.92 | 61.0 | 56.1 |

## Box confidence ĝ·κ̂ vs (present ∧ IoU>=0.5)  [the v0→v1 regression metric]

| set | method | ECE15 | ECE15-mass | Brier | π̂ (EM) |
|---|---|---|---|---|---|
| refcoco_val | raw | 1.68 | 1.75 | 4.87 | - |
| refcoco_val | evalpy_T | 1.66 | 1.72 | 4.87 | - |
| refcoco_val | g:platt|k:platt | 2.15 | 2.19 | 4.82 | - |
| refcoco_val | g:platt+em|k:platt | 0.56 | 0.67 | 4.30 | 0.999 |
| refcoco_val | g:platt|k:iso | 2.22 | 2.26 | 4.84 | - |
| refcoco_val | g:platt+em|k:iso | 0.89 | 0.78 | 4.30 | 0.999 |
| refcoco_val | g:platt|k:multi | 2.76 | 2.76 | 4.82 | - |
| refcoco_val | g:platt+em|k:multi | 1.07 | 0.95 | 4.27 | 0.999 |
| refcoco_val | g:iso|k:platt | 2.08 | 2.10 | 4.84 | - |
| refcoco_val | g:iso+em|k:platt | 0.56 | 0.66 | 4.30 | 0.999 |
| refcoco_val | g:iso|k:iso | 2.15 | 2.22 | 4.85 | - |
| refcoco_val | g:iso+em|k:iso | 0.90 | 0.76 | 4.30 | 0.999 |
| refcoco_val | g:iso|k:multi | 2.69 | 2.69 | 4.84 | - |
| refcoco_val | g:iso+em|k:multi | 1.06 | 0.95 | 4.27 | 0.999 |
| refcoco_val | kappa_only_raw | 0.69 | 0.51 | 4.29 | - |
| refcoco_testA | raw | 1.53 | 1.49 | 3.71 | - |
| refcoco_testA | evalpy_T | 1.52 | 1.48 | 3.71 | - |
| refcoco_testA | g:platt|k:platt | 1.64 | 1.65 | 3.66 | - |
| refcoco_testA | g:platt+em|k:platt | 0.65 | 0.49 | 3.26 | 0.999 |
| refcoco_testA | g:platt|k:iso | 1.52 | 1.50 | 3.68 | - |
| refcoco_testA | g:platt+em|k:iso | 0.66 | 0.41 | 3.27 | 0.999 |
| refcoco_testA | g:platt|k:multi | 2.01 | 2.02 | 3.69 | - |
| refcoco_testA | g:platt+em|k:multi | 0.72 | 0.44 | 3.26 | 0.999 |
| refcoco_testA | g:iso|k:platt | 1.59 | 1.53 | 3.67 | - |
| refcoco_testA | g:iso+em|k:platt | 0.69 | 0.48 | 3.25 | 0.999 |
| refcoco_testA | g:iso|k:iso | 1.42 | 1.51 | 3.69 | - |
| refcoco_testA | g:iso+em|k:iso | 0.70 | 0.40 | 3.26 | 0.999 |
| refcoco_testA | g:iso|k:multi | 1.91 | 1.91 | 3.69 | - |
| refcoco_testA | g:iso+em|k:multi | 0.71 | 0.44 | 3.26 | 0.999 |
| refcoco_testA | kappa_only_raw | 0.87 | 0.87 | 3.27 | - |
| refcoco_testB | raw | 2.44 | 2.44 | 6.07 | - |
| refcoco_testB | evalpy_T | 2.43 | 2.43 | 6.07 | - |
| refcoco_testB | g:platt|k:platt | 3.17 | 3.17 | 6.05 | - |
| refcoco_testB | g:platt+em|k:platt | 1.44 | 1.27 | 5.39 | 0.999 |
| refcoco_testB | g:platt|k:iso | 3.38 | 3.38 | 6.09 | - |
| refcoco_testB | g:platt+em|k:iso | 1.76 | 1.53 | 5.42 | 0.999 |
| refcoco_testB | g:platt|k:multi | 3.75 | 3.75 | 6.09 | - |
| refcoco_testB | g:platt+em|k:multi | 2.04 | 1.87 | 5.41 | 0.999 |
| refcoco_testB | g:iso|k:platt | 3.11 | 3.11 | 6.06 | - |
| refcoco_testB | g:iso+em|k:platt | 1.42 | 1.27 | 5.39 | 0.999 |
| refcoco_testB | g:iso|k:iso | 3.32 | 3.35 | 6.10 | - |
| refcoco_testB | g:iso+em|k:iso | 1.77 | 1.53 | 5.42 | 0.999 |
| refcoco_testB | g:iso|k:multi | 3.70 | 3.69 | 6.10 | - |
| refcoco_testB | g:iso+em|k:multi | 2.04 | 1.87 | 5.41 | 0.999 |
| refcoco_testB | kappa_only_raw | 0.77 | 0.39 | 5.36 | - |
| refcocoplus_val | raw | 1.70 | 2.01 | 6.84 | - |
| refcocoplus_val | evalpy_T | 1.69 | 2.01 | 6.84 | - |
| refcocoplus_val | g:platt|k:platt | 2.26 | 2.50 | 6.79 | - |
| refcocoplus_val | g:platt+em|k:platt | 1.05 | 1.24 | 6.22 | 0.999 |
| refcocoplus_val | g:platt|k:iso | 2.33 | 2.65 | 6.80 | - |
| refcocoplus_val | g:platt+em|k:iso | 1.11 | 0.96 | 6.22 | 0.999 |
| refcocoplus_val | g:platt|k:multi | 2.69 | 2.78 | 6.82 | - |
| refcocoplus_val | g:platt+em|k:multi | 0.63 | 0.85 | 6.23 | 0.999 |
| refcocoplus_val | g:iso|k:platt | 2.20 | 2.50 | 6.80 | - |
| refcocoplus_val | g:iso+em|k:platt | 1.06 | 1.22 | 6.21 | 0.999 |
| refcocoplus_val | g:iso|k:iso | 2.35 | 2.54 | 6.82 | - |
| refcocoplus_val | g:iso+em|k:iso | 1.09 | 0.95 | 6.21 | 0.999 |
| refcocoplus_val | g:iso|k:multi | 2.63 | 2.75 | 6.83 | - |
| refcocoplus_val | g:iso+em|k:multi | 0.61 | 0.82 | 6.22 | 0.999 |
| refcocoplus_val | kappa_only_raw | 1.24 | 1.22 | 6.20 | - |
| refcocoplus_testA | raw | 1.94 | 2.05 | 4.96 | - |
| refcocoplus_testA | evalpy_T | 1.99 | 2.04 | 4.96 | - |
| refcocoplus_testA | g:platt|k:platt | 2.45 | 2.34 | 4.92 | - |
| refcocoplus_testA | g:platt+em|k:platt | 0.94 | 0.92 | 4.42 | 0.999 |
| refcocoplus_testA | g:platt|k:iso | 2.35 | 2.29 | 4.93 | - |
| refcocoplus_testA | g:platt+em|k:iso | 1.08 | 0.92 | 4.42 | 0.999 |
| refcocoplus_testA | g:platt|k:multi | 2.60 | 2.57 | 4.89 | - |
| refcocoplus_testA | g:platt+em|k:multi | 1.52 | 1.32 | 4.36 | 0.999 |
| refcocoplus_testA | g:iso|k:platt | 2.36 | 2.33 | 4.93 | - |
| refcocoplus_testA | g:iso+em|k:platt | 0.93 | 0.92 | 4.41 | 0.999 |
| refcocoplus_testA | g:iso|k:iso | 2.31 | 2.35 | 4.94 | - |
| refcocoplus_testA | g:iso+em|k:iso | 1.08 | 0.92 | 4.42 | 0.999 |
| refcocoplus_testA | g:iso|k:multi | 2.60 | 2.54 | 4.90 | - |
| refcocoplus_testA | g:iso+em|k:multi | 1.53 | 1.29 | 4.36 | 0.999 |
| refcocoplus_testA | kappa_only_raw | 1.24 | 0.97 | 4.42 | - |
| refcocoplus_testB | raw | 2.22 | 2.50 | 9.09 | - |
| refcocoplus_testB | evalpy_T | 2.22 | 2.50 | 9.09 | - |
| refcocoplus_testB | g:platt|k:platt | 2.62 | 3.01 | 9.06 | - |
| refcocoplus_testB | g:platt+em|k:platt | 1.92 | 1.77 | 8.51 | 0.999 |
| refcocoplus_testB | g:platt|k:iso | 2.77 | 3.05 | 9.09 | - |
| refcocoplus_testB | g:platt+em|k:iso | 1.87 | 1.99 | 8.52 | 0.999 |
| refcocoplus_testB | g:platt|k:multi | 2.65 | 2.97 | 9.13 | - |
| refcocoplus_testB | g:platt+em|k:multi | 1.85 | 1.56 | 8.60 | 0.999 |
| refcocoplus_testB | g:iso|k:platt | 2.72 | 3.07 | 9.07 | - |
| refcocoplus_testB | g:iso+em|k:platt | 1.92 | 1.81 | 8.51 | 0.999 |
| refcocoplus_testB | g:iso|k:iso | 3.00 | 3.10 | 9.09 | - |
| refcocoplus_testB | g:iso+em|k:iso | 1.88 | 2.04 | 8.52 | 0.999 |
| refcocoplus_testB | g:iso|k:multi | 2.79 | 3.04 | 9.13 | - |
| refcocoplus_testB | g:iso+em|k:multi | 1.86 | 1.60 | 8.61 | 0.999 |
| refcocoplus_testB | kappa_only_raw | 1.94 | 1.83 | 8.50 | - |
| refcocog_val | raw | 3.80 | 4.11 | 7.61 | - |
| refcocog_val | evalpy_T | 3.79 | 4.10 | 7.61 | - |
| refcocog_val | g:platt|k:platt | 3.90 | 4.28 | 7.50 | - |
| refcocog_val | g:platt+em|k:platt | 1.59 | 1.44 | 6.24 | 0.998 |
| refcocog_val | g:platt|k:iso | 4.26 | 4.58 | 7.53 | - |
| refcocog_val | g:platt+em|k:iso | 1.83 | 1.51 | 6.26 | 0.998 |
| refcocog_val | g:platt|k:multi | 3.85 | 4.16 | 7.50 | - |
| refcocog_val | g:platt+em|k:multi | 1.48 | 1.46 | 6.25 | 0.998 |
| refcocog_val | g:iso|k:platt | 4.17 | 4.36 | 7.52 | - |
| refcocog_val | g:iso+em|k:platt | 1.60 | 1.40 | 6.24 | 0.999 |
| refcocog_val | g:iso|k:iso | 4.52 | 4.59 | 7.55 | - |
| refcocog_val | g:iso+em|k:iso | 1.77 | 1.46 | 6.25 | 0.999 |
| refcocog_val | g:iso|k:multi | 3.87 | 4.32 | 7.52 | - |
| refcocog_val | g:iso+em|k:multi | 1.44 | 1.41 | 6.25 | 0.999 |
| refcocog_val | kappa_only_raw | 1.98 | 1.26 | 6.24 | - |
| refcocog_test | raw | 3.34 | 3.54 | 7.61 | - |
| refcocog_test | evalpy_T | 3.32 | 3.54 | 7.61 | - |
| refcocog_test | g:platt|k:platt | 3.75 | 3.96 | 7.48 | - |
| refcocog_test | g:platt+em|k:platt | 0.91 | 0.92 | 6.12 | 0.997 |
| refcocog_test | g:platt|k:iso | 3.93 | 4.10 | 7.51 | - |
| refcocog_test | g:platt+em|k:iso | 1.23 | 1.01 | 6.15 | 0.997 |
| refcocog_test | g:platt|k:multi | 3.85 | 3.85 | 7.48 | - |
| refcocog_test | g:platt+em|k:multi | 1.03 | 0.66 | 6.12 | 0.997 |
| refcocog_test | g:iso|k:platt | 3.80 | 4.00 | 7.53 | - |
| refcocog_test | g:iso+em|k:platt | 0.82 | 0.87 | 6.10 | 0.998 |
| refcocog_test | g:iso|k:iso | 3.97 | 4.19 | 7.57 | - |
| refcocog_test | g:iso+em|k:iso | 1.22 | 0.97 | 6.12 | 0.998 |
| refcocog_test | g:iso|k:multi | 3.89 | 3.91 | 7.53 | - |
| refcocog_test | g:iso+em|k:multi | 1.03 | 0.64 | 6.09 | 0.998 |
| refcocog_test | kappa_only_raw | 1.51 | 1.23 | 6.10 | - |
| grec_val | raw | 5.30 | 5.31 | 6.60 | - |
| grec_val | evalpy_T | 5.30 | 5.31 | 6.60 | - |
| grec_val | g:platt|k:platt | 5.26 | 5.26 | 6.68 | - |
| grec_val | g:platt+em|k:platt | 3.93 | 3.96 | 6.18 | 0.604 |
| grec_val | g:platt|k:iso | 5.20 | 5.20 | 6.66 | - |
| grec_val | g:platt+em|k:iso | 3.88 | 3.88 | 6.17 | 0.604 |
| grec_val | g:platt|k:multi | 4.83 | 4.83 | 6.58 | - |
| grec_val | g:platt+em|k:multi | 3.52 | 3.54 | 6.10 | 0.604 |
| grec_val | g:iso|k:platt | 5.24 | 5.24 | 6.68 | - |
| grec_val | g:iso+em|k:platt | 3.96 | 3.97 | 6.19 | 0.604 |
| grec_val | g:iso|k:iso | 5.18 | 5.18 | 6.66 | - |
| grec_val | g:iso+em|k:iso | 3.91 | 3.91 | 6.17 | 0.604 |
| grec_val | g:iso|k:multi | 4.81 | 4.81 | 6.58 | - |
| grec_val | g:iso+em|k:multi | 3.55 | 3.55 | 6.11 | 0.604 |
| grec_val | kappa_only_raw | 24.16 | 24.16 | 19.90 | - |
| grec_testA | raw | 7.77 | 7.92 | 8.12 | - |
| grec_testA | evalpy_T | 7.77 | 7.93 | 8.12 | - |
| grec_testA | g:platt|k:platt | 7.79 | 7.93 | 8.19 | - |
| grec_testA | g:platt+em|k:platt | 7.00 | 7.15 | 7.81 | 0.655 |
| grec_testA | g:platt|k:iso | 7.73 | 7.84 | 8.17 | - |
| grec_testA | g:platt+em|k:iso | 6.95 | 7.07 | 7.80 | 0.655 |
| grec_testA | g:platt|k:multi | 7.17 | 7.28 | 7.95 | - |
| grec_testA | g:platt+em|k:multi | 6.41 | 6.53 | 7.59 | 0.655 |
| grec_testA | g:iso|k:platt | 7.78 | 7.84 | 8.18 | - |
| grec_testA | g:iso+em|k:platt | 7.02 | 7.11 | 7.81 | 0.655 |
| grec_testA | g:iso|k:iso | 7.72 | 7.80 | 8.17 | - |
| grec_testA | g:iso+em|k:iso | 6.97 | 7.07 | 7.79 | 0.655 |
| grec_testA | g:iso|k:multi | 7.17 | 7.22 | 7.94 | - |
| grec_testA | g:iso+em|k:multi | 6.43 | 6.51 | 7.59 | 0.655 |
| grec_testA | kappa_only_raw | 25.05 | 25.05 | 21.36 | - |
| grec_testB | raw | 9.46 | 9.62 | 11.18 | - |
| grec_testB | evalpy_T | 9.46 | 9.62 | 11.19 | - |
| grec_testB | g:platt|k:platt | 9.26 | 9.39 | 11.25 | - |
| grec_testB | g:platt+em|k:platt | 7.85 | 7.95 | 10.53 | 0.6 |
| grec_testB | g:platt|k:iso | 9.14 | 9.21 | 11.20 | - |
| grec_testB | g:platt+em|k:iso | 7.74 | 7.74 | 10.49 | 0.6 |
| grec_testB | g:platt|k:multi | 8.79 | 8.89 | 11.10 | - |
| grec_testB | g:platt+em|k:multi | 7.40 | 7.48 | 10.41 | 0.6 |
| grec_testB | g:iso|k:platt | 9.09 | 9.16 | 11.23 | - |
| grec_testB | g:iso+em|k:platt | 7.87 | 7.88 | 10.50 | 0.6 |
| grec_testB | g:iso|k:iso | 9.04 | 9.02 | 11.18 | - |
| grec_testB | g:iso+em|k:iso | 7.77 | 7.79 | 10.47 | 0.6 |
| grec_testB | g:iso|k:multi | 8.65 | 8.67 | 11.08 | - |
| grec_testB | g:iso+em|k:multi | 7.45 | 7.46 | 10.38 | 0.6 |
| grec_testB | kappa_only_raw | 26.98 | 26.98 | 23.85 | - |

Bootstrap (image clusters): 
- refcoco_val: raw ECE 1.68 [1.28, 2.24]; fixed 0.56 [0.51, 1.19]; paired diff -1.12 [-1.32, -0.48]; κ̂ AUROC 90.9 [89.5, 92.3]
- refcocog_test: raw ECE 3.34 [2.89, 3.97]; fixed 0.91 [0.81, 1.57]; paired diff -2.43 [-2.92, -1.57]; κ̂ AUROC 88.5 [87.3, 89.7]

## Presence bootstrap CIs (image clusters, raw ĝ unless noted)

- grec_val: AUROC 96.9 [96.7, 97.2]; N-acc 83.8 [83.0, 84.6]; ECE raw 3.25 [2.97, 3.59] → Platt 3.83 [3.54, 4.18] (paired 0.58 [0.47, 0.64]); subset ECE raw 7.39 [6.87, 7.92] → Platt 8.28 [7.75, 8.84]
- grec_testA: AUROC 91.6 [90.9, 92.1]; N-acc 74.3 [72.9, 75.6]; ECE raw 5.58 [5.12, 6.21] → Platt 5.52 [5.02, 6.15] (paired -0.06 [-0.26, 0.04]); subset ECE raw 5.58 [5.12, 6.21] → Platt 5.52 [5.02, 6.15]
- grec_testB: AUROC 87.2 [85.8, 88.7]; N-acc 71.6 [70.2, 73.0]; ECE raw 9.04 [7.90, 10.32] → Platt 9.18 [8.00, 10.50] (paired 0.15 [-0.06, 0.36]); subset ECE raw 9.04 [7.90, 10.32] → Platt 9.18 [8.00, 10.50]
- venus: AUROC 79.3 [77.7, 80.9]; N-acc 83.4 [80.9, 86.1]; ECE raw 32.75 [31.26, 34.15] → Platt 24.88 [23.78, 25.93] (paired -7.88 [-8.52, -7.38]); subset ECE raw 32.75 [31.26, 34.15] → Platt 24.88 [23.78, 25.93]

## Risk-controlled accept/abstain (δ=0.1; 50/50 image splits ×reps; coverage = fraction answered)


**refcoco_val** (n=10834, absent 0.0%, correct-if-answered-all 93.5%); default ĝ≥.5: coverage 98.9, selective risk 6.3, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 28.9 | 1.11 | 0.06 |
| selective[g] | 0.05 | 82.3 | 4.18 | 0.08 |
| selective[g] | 0.1 | 100.0 | 6.48 | 0.0 |
| joint[g] | 0.01 | 46.6 | 0.77 | 0.18 |
| joint[g] | 0.02 | 67.2 | 1.78 | 0.29 |
| joint[g] | 0.05 | 89.7 | 4.56 | 0.22 |
| joint[g] | 0.1 | 100.0 | 6.50 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 76.4 | 3.47 | False |
| selective_transfer[g] | 0.1 | 97.1 | 6.12 | False |
| selective[g*k] | 0.01 | 52.4 | 0.56 | 0.05 |
| selective[g*k] | 0.02 | 79.2 | 1.54 | 0.13 |
| selective[g*k] | 0.05 | 95.1 | 4.25 | 0.12 |
| selective[g*k] | 0.1 | 100.0 | 6.47 | 0.0 |
| joint[g*k] | 0.01 | 73.0 | 0.79 | 0.15 |
| joint[g*k] | 0.02 | 84.7 | 1.76 | 0.25 |
| joint[g*k] | 0.05 | 96.3 | 4.54 | 0.22 |
| joint[g*k] | 0.1 | 100.0 | 6.50 | 0.0 |
| selective_transfer[g*k] | 0.01 | 53.7 | 0.48 | False |
| selective_transfer[g*k] | 0.02 | 74.7 | 1.21 | False |
| selective_transfer[g*k] | 0.05 | 89.7 | 2.73 | False |
| selective_transfer[g*k] | 0.1 | 97.7 | 5.31 | False |

AURC ĝ 2.29 vs ĝκ̂ 1.02; paired diff -1.27 [-1.53, -1.02]

**refcocog_test** (n=9602, absent 0.0%, correct-if-answered-all 91.0%); default ĝ≥.5: coverage 96.8, selective risk 8.6, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 4.8 | 0.41 | 0.06 |
| selective[g] | 0.05 | 57.5 | 4.05 | 0.08 |
| selective[g] | 0.1 | 98.7 | 8.84 | 0.0 |
| joint[g] | 0.01 | 36.2 | 0.78 | 0.15 |
| joint[g] | 0.02 | 50.7 | 1.74 | 0.18 |
| joint[g] | 0.05 | 75.6 | 4.55 | 0.14 |
| joint[g] | 0.1 | 99.9 | 9.01 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 55.6 | 3.84 | False |
| selective_transfer[g] | 0.1 | 92.7 | 8.08 | False |
| selective[g*k] | 0.01 | 6.4 | 0.15 | 0.02 |
| selective[g*k] | 0.02 | 55.3 | 1.37 | 0.05 |
| selective[g*k] | 0.05 | 84.6 | 4.26 | 0.08 |
| selective[g*k] | 0.1 | 99.4 | 8.83 | 0.0 |
| joint[g*k] | 0.01 | 56.1 | 0.76 | 0.07 |
| joint[g*k] | 0.02 | 70.8 | 1.72 | 0.17 |
| joint[g*k] | 0.05 | 88.4 | 4.58 | 0.18 |
| joint[g*k] | 0.1 | 99.9 | 9.02 | 0.0 |
| selective_transfer[g*k] | 0.01 | 42.6 | 0.86 | False |
| selective_transfer[g*k] | 0.02 | 64.6 | 1.87 | False |
| selective_transfer[g*k] | 0.05 | 84.8 | 4.22 | False |
| selective_transfer[g*k] | 0.1 | 96.0 | 7.59 | False |

AURC ĝ 3.82 vs ĝκ̂ 2.04; paired diff -1.78 [-2.03, -1.53]

**grec_val** (n=25063, absent 35.5%, correct-if-answered-all 56.9%); default ĝ≥.5: coverage 68.3, selective risk 18.7, false-accept on absent 16.2

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 4.3 | 0.78 | 0.09 |
| selective[g] | 0.05 | 30.9 | 4.11 | 0.04 |
| selective[g] | 0.1 | 46.7 | 9.06 | 0.1 |
| joint[g] | 0.01 | 25.7 | 0.87 | 0.15 |
| joint[g] | 0.02 | 35.5 | 1.83 | 0.21 |
| joint[g] | 0.05 | 48.4 | 4.67 | 0.17 |
| joint[g] | 0.1 | 62.5 | 9.53 | 0.16 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 38.2 | 5.88 | True |
| selective_transfer[g] | 0.1 | 63.1 | 15.57 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 5.3 | 0.66 | 0.01 |
| selective[g*k] | 0.05 | 38.9 | 4.26 | 0.07 |
| selective[g*k] | 0.1 | 52.5 | 9.11 | 0.07 |
| joint[g*k] | 0.01 | 30.8 | 0.88 | 0.18 |
| joint[g*k] | 0.02 | 40.2 | 1.83 | 0.24 |
| joint[g*k] | 0.05 | 52.2 | 4.67 | 0.09 |
| joint[g*k] | 0.1 | 63.4 | 9.48 | 0.13 |
| selective_transfer[g*k] | 0.01 | 25.5 | 2.14 | True |
| selective_transfer[g*k] | 0.02 | 38.7 | 4.26 | True |
| selective_transfer[g*k] | 0.05 | 53.9 | 9.78 | True |
| selective_transfer[g*k] | 0.1 | 67.3 | 17.61 | True |
| absent_conformal[g] | 0.01 | present accepted 46.4 | false-accept absent 1.00 | |
| absent_conformal[g] | 0.02 | present accepted 62.6 | false-accept absent 1.99 | |
| absent_conformal[g] | 0.05 | present accepted 83.1 | false-accept absent 5.11 | |
| absent_conformal[g] | 0.1 | present accepted 92.8 | false-accept absent 10.01 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 65.6 | false-accept absent 2.28 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 85.9 | false-accept absent 6.02 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 95.9 | false-accept absent 14.17 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 98.6 | false-accept absent 23.07 | |

AURC ĝ 14.60 vs ĝκ̂ 13.80; paired diff -0.81 [-1.01, -0.63]

**grec_testA** (n=19200, absent 23.2%, correct-if-answered-all 58.8%); default ĝ≥.5: coverage 77.7, selective risk 27.2, false-accept on absent 25.7

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.8 | 0.72 | 0.08 |
| selective[g] | 0.1 | 25.7 | 8.62 | 0.15 |
| joint[g] | 0.01 | 14.8 | 0.85 | 0.17 |
| joint[g] | 0.02 | 23.4 | 1.78 | 0.23 |
| joint[g] | 0.05 | 36.7 | 4.71 | 0.32 |
| joint[g] | 0.1 | 51.9 | 9.64 | 0.31 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 38.7 | 13.65 | True |
| selective_transfer[g] | 0.1 | 71.3 | 24.63 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 26.0 | 3.88 | 0.04 |
| selective[g*k] | 0.1 | 39.7 | 8.96 | 0.16 |
| joint[g*k] | 0.01 | 24.2 | 0.87 | 0.12 |
| joint[g*k] | 0.02 | 31.9 | 1.76 | 0.17 |
| joint[g*k] | 0.05 | 43.4 | 4.66 | 0.24 |
| joint[g*k] | 0.1 | 56.9 | 9.58 | 0.33 |
| selective_transfer[g*k] | 0.01 | 24.7 | 3.61 | True |
| selective_transfer[g*k] | 0.02 | 39.0 | 8.47 | True |
| selective_transfer[g*k] | 0.05 | 58.8 | 17.52 | True |
| selective_transfer[g*k] | 0.1 | 75.1 | 24.80 | True |
| absent_conformal[g] | 0.01 | present accepted 17.7 | false-accept absent 0.99 | |
| absent_conformal[g] | 0.02 | present accepted 30.5 | false-accept absent 2.05 | |
| absent_conformal[g] | 0.05 | present accepted 53.4 | false-accept absent 4.82 | |
| absent_conformal[g] | 0.1 | present accepted 75.5 | false-accept absent 10.01 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 55.9 | false-accept absent 5.26 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 79.8 | false-accept absent 11.74 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 91.9 | false-accept absent 23.18 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 95.8 | false-accept absent 32.82 | |

AURC ĝ 18.32 vs ĝκ̂ 15.75; paired diff -2.56 [-2.90, -2.20]

**grec_testB** (n=16063, absent 29.1%, correct-if-answered-all 50.6%); default ĝ≥.5: coverage 72.5, selective risk 33.1, false-accept on absent 28.4

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 6.9 | 0.86 | 0.25 |
| joint[g] | 0.02 | 13.0 | 1.80 | 0.27 |
| joint[g] | 0.05 | 27.5 | 4.58 | 0.21 |
| joint[g] | 0.1 | 42.4 | 9.47 | 0.3 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 32.0 | 18.14 | True |
| selective_transfer[g] | 0.1 | 66.1 | 30.06 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.1 | 0.06 | 0.01 |
| selective[g*k] | 0.1 | 19.5 | 8.11 | 0.09 |
| joint[g*k] | 0.01 | 14.3 | 0.88 | 0.26 |
| joint[g*k] | 0.02 | 20.7 | 1.80 | 0.22 |
| joint[g*k] | 0.05 | 31.8 | 4.59 | 0.22 |
| joint[g*k] | 0.1 | 45.8 | 9.45 | 0.2 |
| selective_transfer[g*k] | 0.01 | 16.6 | 6.84 | True |
| selective_transfer[g*k] | 0.02 | 29.9 | 13.79 | True |
| selective_transfer[g*k] | 0.05 | 50.8 | 22.60 | True |
| selective_transfer[g*k] | 0.1 | 69.3 | 30.69 | True |
| absent_conformal[g] | 0.01 | present accepted 9.0 | false-accept absent 1.01 | |
| absent_conformal[g] | 0.02 | present accepted 18.7 | false-accept absent 1.99 | |
| absent_conformal[g] | 0.05 | present accepted 43.7 | false-accept absent 4.97 | |
| absent_conformal[g] | 0.1 | present accepted 66.7 | false-accept absent 9.89 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 51.6 | false-accept absent 6.23 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 77.1 | false-accept absent 14.00 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 89.4 | false-accept absent 25.98 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 92.6 | false-accept absent 36.19 | |

AURC ĝ 26.42 vs ĝκ̂ 23.25; paired diff -3.17 [-3.90, -2.52]

**ss2** (n=1272, absent 0.0%, correct-if-answered-all 89.9%); default ĝ≥.5: coverage 90.0, selective risk 9.1, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 71.5 | 6.75 | 0.05 |
| joint[g] | 0.01 | 9.6 | 0.42 | 0.06 |
| joint[g] | 0.02 | 29.8 | 1.18 | 0.1 |
| joint[g] | 0.05 | 66.2 | 3.96 | 0.13 |
| joint[g] | 0.1 | 90.6 | 8.45 | 0.18 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 1.7 | 9.09 | True |
| selective_transfer[g] | 0.1 | 12.8 | 3.68 | False |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 12.5 | 0.90 | 0.03 |
| selective[g*k] | 0.1 | 85.3 | 7.05 | 0.07 |
| joint[g*k] | 0.01 | 28.5 | 0.49 | 0.12 |
| joint[g*k] | 0.02 | 43.4 | 1.18 | 0.08 |
| joint[g*k] | 0.05 | 71.8 | 3.96 | 0.2 |
| joint[g*k] | 0.1 | 95.3 | 8.45 | 0.2 |
| selective_transfer[g*k] | 0.01 | 0.4 | 0.00 | False |
| selective_transfer[g*k] | 0.02 | 8.8 | 0.00 | False |
| selective_transfer[g*k] | 0.05 | 61.1 | 3.35 | False |
| selective_transfer[g*k] | 0.1 | 91.3 | 7.75 | False |

AURC ĝ 5.81 vs ĝκ̂ 3.55; paired diff -2.26 [-3.41, -1.29]

**venus** (n=6165, absent 14.6%, correct-if-answered-all 47.9%); default ĝ≥.5: coverage 53.9, selective risk 37.2, false-accept on absent 16.6

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 4.3 | 0.81 | 0.19 |
| joint[g] | 0.02 | 7.7 | 1.76 | 0.26 |
| joint[g] | 0.05 | 17.9 | 4.47 | 0.18 |
| joint[g] | 0.1 | 31.0 | 9.39 | 0.23 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 50.00 | True |
| selective_transfer[g] | 0.1 | 1.2 | 11.69 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 0.1 | 0.43 | 0.03 |
| joint[g*k] | 0.01 | 6.2 | 0.78 | 0.2 |
| joint[g*k] | 0.02 | 11.5 | 1.74 | 0.22 |
| joint[g*k] | 0.05 | 24.0 | 4.40 | 0.13 |
| joint[g*k] | 0.1 | 37.7 | 9.22 | 0.18 |
| selective_transfer[g*k] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.02 | 1.2 | 1.37 | False |
| selective_transfer[g*k] | 0.05 | 23.7 | 18.13 | True |
| selective_transfer[g*k] | 0.1 | 51.9 | 30.39 | True |
| absent_conformal[g] | 0.01 | present accepted 13.6 | false-accept absent 0.97 | |
| absent_conformal[g] | 0.02 | present accepted 21.6 | false-accept absent 1.98 | |
| absent_conformal[g] | 0.05 | present accepted 34.4 | false-accept absent 4.91 | |
| absent_conformal[g] | 0.1 | present accepted 47.3 | false-accept absent 9.68 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 0.4 | false-accept absent 0.00 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 0.8 | false-accept absent 0.00 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 2.8 | false-accept absent 0.11 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 8.6 | false-accept absent 0.33 | |

AURC ĝ 35.75 vs ĝκ̂ 29.78; paired diff -5.97 [-6.83, -5.29]

## Prop. 1 two-knob plug-in rule vs one-knob thresholds (expected cost per query, lower is better)

| set | costs | two-knob plug-in | ĝ≥.5 | ĝ thr tuned on calib | ĝκ̂ thr tuned | ĝ oracle thr | paired two−one(tuned) |
|---|---|---|---|---|---|---|---|
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0672 | 0.0680 | 0.0693 | 0.0666 | 0.0646 | -0.0021 [-0.0041, -0.0002] |
| refcoco_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1005 | 0.1304 | 0.1372 | 0.0995 | 0.1287 | -0.0367 [-0.0434, -0.0302] |
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.0743 | 0.0736 | 0.0744 | 0.0733 | 0.0650 | -0.0001 [-0.0009, 0.0007] |
| refcoco_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0704 | 0.0680 | 0.0742 | 0.0666 | 0.0646 | -0.0038 [-0.0058, -0.0017] |
| refcoco_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1435 | 0.2552 | 0.2417 | 0.1496 | 0.2417 | -0.0982 [-0.1161, -0.0816] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0997 | 0.0993 | 0.1016 | 0.0981 | 0.0911 | -0.0020 [-0.0044, 0.0003] |
| refcocog_test | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1453 | 0.1825 | 0.2002 | 0.1490 | 0.1794 | -0.0549 [-0.0626, -0.0485] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1159 | 0.1153 | 0.1161 | 0.1100 | 0.0930 | -0.0002 [-0.0012, 0.0008] |
| refcocog_test | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1078 | 0.0993 | 0.1117 | 0.0981 | 0.0911 | -0.0039 [-0.0066, -0.0018] |
| refcocog_test | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2065 | 0.3489 | 0.3214 | 0.2119 | 0.3214 | -0.1150 [-0.1315, -0.1000] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1165 | 0.1377 | 0.1314 | 0.1189 | 0.1199 | -0.0149 [-0.0171, -0.0127] |
| grec_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1457 | 0.2082 | 0.1709 | 0.1476 | 0.1709 | -0.0252 [-0.0297, -0.0204] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1453 | 0.1474 | 0.1472 | 0.1504 | 0.1444 | -0.0020 [-0.0032, -0.0010] |
| grec_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1406 | 0.1953 | 0.1563 | 0.1560 | 0.1397 | -0.0157 [-0.0182, -0.0127] |
| grec_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1706 | 0.3490 | 0.2719 | 0.1674 | 0.2719 | -0.1012 [-0.1118, -0.0890] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2080 | 0.2365 | 0.2310 | 0.2091 | 0.2194 | -0.0230 [-0.0261, -0.0198] |
| grec_testA | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2774 | 0.3879 | 0.3290 | 0.2894 | 0.3290 | -0.0516 [-0.0594, -0.0431] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2572 | 0.2621 | 0.2619 | 0.2550 | 0.2614 | -0.0046 [-0.0062, -0.0028] |
| grec_testA | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2353 | 0.2959 | 0.2628 | 0.2461 | 0.2448 | -0.0275 [-0.0317, -0.0235] |
| grec_testA | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.3430 | 0.6909 | 0.5475 | 0.3221 | 0.5475 | -0.2046 [-0.2271, -0.1837] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2410 | 0.2735 | 0.2661 | 0.2411 | 0.2416 | -0.0251 [-0.0290, -0.0215] |
| grec_testB | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.3059 | 0.4312 | 0.3578 | 0.3153 | 0.3578 | -0.0519 [-0.0622, -0.0424] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.3013 | 0.3067 | 0.3063 | 0.3065 | 0.3024 | -0.0050 [-0.0069, -0.0031] |
| grec_testB | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2815 | 0.3562 | 0.3087 | 0.2995 | 0.2793 | -0.0272 [-0.0322, -0.0223] |
| grec_testB | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.3595 | 0.7466 | 0.5901 | 0.3413 | 0.5901 | -0.2306 [-0.2566, -0.2019] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1726 | 0.1317 | 0.1580 | 0.1380 | 0.1010 | 0.0145 [0.0052, 0.0244] |
| ss2 | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2445 | 0.2134 | 0.2256 | 0.2127 | 0.2001 | 0.0189 [0.0006, 0.0404] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1627 | 0.1816 | 0.1737 | 0.1517 | 0.1022 | -0.0110 [-0.0187, -0.0039] |
| ss2 | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2205 | 0.1317 | 0.1698 | 0.1474 | 0.1010 | 0.0507 [0.0405, 0.0633] |
| ss2 | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.3223 | 0.3770 | 0.3274 | 0.2932 | 0.3204 | -0.0051 [-0.0459, 0.0320] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.3467 | 0.3700 | 0.3655 | 0.3392 | 0.3646 | -0.0187 [-0.0242, -0.0135] |
| venus | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.3912 | 0.5460 | 0.4651 | 0.4005 | 0.4202 | -0.0740 [-0.0877, -0.0628] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.5330 | 0.5398 | 0.5283 | 0.5267 | 0.4863 | 0.0047 [-0.0005, 0.0095] |
| venus | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.3729 | 0.3942 | 0.3774 | 0.3593 | 0.3756 | -0.0045 [-0.0110, 0.0016] |
| venus | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.4290 | 0.8980 | 0.5558 | 0.4492 | 0.4306 | -0.1268 [-0.1487, -0.1066] |
