# sprint_abs tables (v1.1)

## GREC official protocol (gRefCOCO; Pr@(F1=1, GIoU>=0.5), N-acc, T-acc)

| split | variant | Pr@F1=1 | 95% CI | N-acc | T-acc | Pr single | Pr multi | Pr (plain IoU) |
|---|---|---|---|---|---|---|---|---|
| grec_val | single_raw@0.7(official) | 72.2 | 72.2 [71.7, 72.7] | 89.6 | 96.8 | 93.3 | 0.0 | 72.3 |
| grec_val | single_raw@0.5 | 71.1 | 71.1 [70.6, 71.6] | 86.0 | 98.2 | 93.8 | 0.0 | 71.2 |
| grec_val | single_platt@0.7 | 72.3 | 72.3 [71.8, 72.8] | 90.3 | 96.3 | 93.1 | 0.0 | 72.5 |
| grec_val | multi_platt@0.7(official) | 72.3 | 72.3 [71.8, 72.8] | 90.3 | 96.3 | 93.1 | 0.0 | 72.5 |
| grec_val | multi_raw@0.7(official) | 72.2 | 72.2 [71.7, 72.7] | 89.6 | 96.8 | 93.3 | 0.1 | 72.3 |
| grec_val | multi_platt@tuned | 71.5 | 71.5 [71.0, 72.0] | 84.2 | 98.6 | 93.4 | 5.8 | 71.8 |
| grec_val | single_platt@tuned | 71.4 | 71.4 [70.9, 71.9] | 87.0 | 97.9 | 93.7 | 0.0 | 71.6 |
| grec_val | paired diff multi-single (official) | 0.2 [0.1, 0.2] | | | | | | |
| grec_testA | single_raw@0.7(official) | 47.6 | 47.6 [46.8, 48.3] | 82.4 | 94.1 | 92.4 | 0.0 | 47.7 |
| grec_testA | single_raw@0.5 | 46.9 | 46.9 [46.2, 47.7] | 79.3 | 95.7 | 92.7 | 0.0 | 47.0 |
| grec_testA | single_platt@0.7 | 47.7 | 47.7 [47.0, 48.4] | 83.2 | 93.4 | 92.2 | 0.0 | 47.8 |
| grec_testA | multi_platt@0.7(official) | 47.7 | 47.7 [47.0, 48.5] | 83.2 | 93.4 | 92.2 | 0.1 | 47.8 |
| grec_testA | multi_raw@0.7(official) | 47.6 | 47.6 [46.9, 48.4] | 82.4 | 94.1 | 92.4 | 0.1 | 47.7 |
| grec_testA | multi_platt@tuned | 48.2 | 48.2 [47.4, 49.0] | 77.6 | 96.2 | 92.4 | 3.8 | 48.4 |
| grec_testA | single_platt@tuned | 47.1 | 47.1 [46.4, 47.8] | 80.1 | 95.4 | 92.6 | 0.0 | 47.2 |
| grec_testA | paired diff multi-single (official) | 0.2 [0.1, 0.3] | | | | | | |
| grec_testB | single_raw@0.7(official) | 51.5 | 51.5 [50.4, 52.5] | 77.9 | 89.7 | 82.0 | 0.0 | 51.6 |
| grec_testB | single_raw@0.5 | 50.5 | 50.5 [49.5, 51.6] | 74.0 | 91.7 | 82.5 | 0.0 | 50.7 |
| grec_testB | single_platt@0.7 | 51.7 | 51.7 [50.6, 52.8] | 78.9 | 89.0 | 81.8 | 0.0 | 51.9 |
| grec_testB | multi_platt@0.7(official) | 51.7 | 51.7 [50.7, 52.8] | 78.9 | 89.0 | 81.8 | 0.1 | 51.9 |
| grec_testB | multi_raw@0.7(official) | 51.5 | 51.5 [50.5, 52.6] | 77.9 | 89.7 | 82.0 | 0.1 | 51.7 |
| grec_testB | multi_platt@tuned | 51.0 | 51.0 [50.0, 52.1] | 72.1 | 92.4 | 82.2 | 3.1 | 51.2 |
| grec_testB | single_platt@tuned | 50.8 | 50.8 [49.8, 51.8] | 74.9 | 91.3 | 82.5 | 0.0 | 50.9 |
| grec_testB | paired diff multi-single (official) | 0.3 [0.2, 0.4] | | | | | | |
| grefcalib | single_raw@0.7(official) | 64.4 | - | 98.0 | 97.7 | 93.2 | 0.0 | 64.7 |
| grefcalib | single_raw@0.5 | 64.5 | - | 96.7 | 98.7 | 93.7 | 0.0 | 64.8 |
| grefcalib | single_platt@0.7 | 64.2 | - | 98.2 | 97.3 | 92.9 | 0.0 | 64.5 |
| grefcalib | multi_platt@0.7(official) | 64.3 | - | 98.2 | 97.3 | 92.9 | 0.1 | 64.5 |
| grefcalib | multi_raw@0.7(official) | 64.5 | - | 98.0 | 97.7 | 93.2 | 0.1 | 64.7 |
| grefcalib | multi_platt@tuned | 66.0 | - | 96.0 | 99.0 | 93.0 | 6.0 | 66.5 |
| grefcalib | single_platt@tuned | 64.6 | - | 97.3 | 98.4 | 93.6 | 0.0 | 64.8 |

Tuned on grefcalib: {"pr_f1_grefcalib": 0.6601052872510872, "t_presence": 0.35, "t_extra": 0.3, "single_t_presence": 0.49999999999999994, "single_pr_f1_grefcalib": 0.6456855115587091}; extra-box model: {"n": 1148, "pos": 0.2229965156794425}

## VenusBench-GD, official UI-Venus evaluator

| config | Element | Visual | Spatial | Basic | Reasoning | Functional | Refusal | Advanced | Overall | n |
|---|---|---|---|---|---|---|---|---|---|---|
| platt_gated | 52.55 | 61.38 | 46.55 | 52.91 | 14.72 | 37.57 | 63.85 | 36.95 | 45.90 | 6165 |
| raw_gated | 43.49 | 42.07 | 35.28 | 40.71 | 10.93 | 32.29 | 83.76 | 40.65 | 40.68 | 6165 |
| raw_ungated | 59.33 | 77.23 | 59.57 | 63.75 | 17.43 | 40.43 | 83.76 | 45.42 | 55.70 | 6165 |
| stack_gated | 55.44 | 74.85 | 56.17 | 60.36 | 9.58 | 38.43 | 36.26 | 25.91 | 45.24 | 6165 |

## Presence ĝ calibration (fit on held-out training-distribution splits)

| set | subset | method | AUROC | N-acc@.5 | recall@.5 | ECE15 | ECE15-mass | ECE10 | Brier | NLL | +EM prior: ECE15 / N-acc / π̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grec_val | all | raw | 98.0 | 86.0 | 98.2 | 3.6 | 3.6 | 3.6 | 4.84 | 0.190 | - |
| grec_val | all | temp | 98.0 | 86.0 | 98.2 | 3.6 | 3.6 | 3.6 | 4.83 | 0.190 | 3.0 / 87.1 / 0.674 |
| grec_val | all | platt | 98.0 | 87.0 | 97.9 | 3.0 | 3.0 | 3.0 | 4.70 | 0.184 | 2.3 / 88.2 / 0.667 |
| grec_val | all | beta | 98.0 | 86.2 | 98.1 | 3.3 | 3.2 | 3.2 | 4.77 | 0.181 | 2.6 / 87.3 / 0.668 |
| grec_val | all | iso | 98.0 | 85.4 | 98.3 | 3.2 | 3.2 | 3.1 | 4.82 | 0.184 | 2.7 / 87.1 / 0.669 |
| grec_val | all | multi | 98.0 | 86.9 | 97.8 | 2.7 | 2.7 | 2.7 | 4.66 | 0.170 | 1.6 / 88.7 / 0.660 |
| grec_val | gref_only | raw | 96.2 | 86.0 | 96.2 | 7.5 | 7.5 | 7.5 | 8.06 | 0.318 | - |
| grec_val | gref_only | temp | 96.2 | 86.0 | 96.2 | 7.5 | 7.5 | 7.5 | 8.06 | 0.317 | 3.2 / 91.5 / 0.386 |
| grec_val | gref_only | platt | 96.2 | 87.0 | 95.5 | 6.5 | 6.5 | 6.5 | 7.74 | 0.304 | 3.6 / 92.3 / 0.375 |
| grec_val | gref_only | beta | 96.2 | 86.2 | 96.0 | 6.8 | 6.9 | 6.7 | 7.92 | 0.300 | 3.0 / 91.7 / 0.377 |
| grec_val | gref_only | iso | 96.2 | 85.4 | 96.4 | 6.7 | 6.8 | 6.7 | 7.98 | 0.304 | 3.3 / 91.4 / 0.382 |
| grec_val | gref_only | multi | 96.1 | 86.9 | 95.5 | 6.5 | 6.5 | 6.5 | 7.64 | 0.276 | 4.2 / 93.6 / 0.343 |
| grec_testA | all | raw | 93.7 | 79.3 | 95.7 | 4.5 | 4.3 | 4.5 | 6.71 | 0.310 | - |
| grec_testA | all | temp | 93.7 | 79.3 | 95.7 | 4.5 | 4.3 | 4.5 | 6.70 | 0.309 | 4.5 / 77.9 / 0.785 |
| grec_testA | all | platt | 93.7 | 80.1 | 95.4 | 4.7 | 4.6 | 4.7 | 6.77 | 0.311 | 4.5 / 79.0 / 0.778 |
| grec_testA | all | beta | 93.7 | 79.5 | 95.6 | 4.6 | 4.4 | 4.6 | 6.76 | 0.317 | 4.7 / 78.3 / 0.780 |
| grec_testA | all | iso | 93.7 | 78.8 | 95.9 | 4.9 | 4.6 | 4.9 | 6.87 | 0.304 | 5.0 / 78.3 / 0.780 |
| grec_testA | all | multi | 93.6 | 80.1 | 95.3 | 3.6 | 3.6 | 3.6 | 6.70 | 0.271 | 3.4 / 78.5 / 0.774 |
| grec_testA | gref_only | raw | 92.5 | 79.3 | 95.0 | 5.3 | 5.2 | 5.3 | 8.36 | 0.360 | - |
| grec_testA | gref_only | temp | 92.5 | 79.3 | 95.0 | 5.3 | 5.2 | 5.3 | 8.35 | 0.359 | 5.4 / 80.0 / 0.686 |
| grec_testA | gref_only | platt | 92.5 | 80.1 | 94.6 | 5.5 | 5.4 | 5.4 | 8.40 | 0.357 | 5.7 / 80.9 / 0.676 |
| grec_testA | gref_only | beta | 92.5 | 79.5 | 95.0 | 5.5 | 5.3 | 5.4 | 8.40 | 0.349 | 5.4 / 80.1 / 0.679 |
| grec_testA | gref_only | iso | 92.5 | 78.8 | 95.3 | 5.9 | 5.8 | 5.8 | 8.54 | 0.350 | 5.9 / 79.1 / 0.681 |
| grec_testA | gref_only | multi | 92.4 | 80.1 | 94.5 | 4.0 | 3.8 | 4.0 | 8.31 | 0.317 | 4.5 / 81.4 / 0.663 |
| grec_testB | all | raw | 88.2 | 74.0 | 91.7 | 9.2 | 9.3 | 9.2 | 11.66 | 0.632 | - |
| grec_testB | all | temp | 88.2 | 74.0 | 91.7 | 9.2 | 9.3 | 9.2 | 11.66 | 0.629 | 9.2 / 74.1 / 0.717 |
| grec_testB | all | platt | 88.2 | 74.9 | 91.3 | 9.2 | 9.1 | 9.1 | 11.66 | 0.636 | 9.2 / 75.3 / 0.709 |
| grec_testB | all | beta | 88.2 | 74.2 | 91.6 | 9.3 | 9.4 | 9.3 | 11.69 | 0.683 | 9.3 / 74.4 / 0.711 |
| grec_testB | all | iso | 88.3 | 73.4 | 91.9 | 9.7 | 9.8 | 9.7 | 11.80 | 0.617 | 9.7 / 73.5 / 0.712 |
| grec_testB | all | multi | 88.1 | 74.9 | 91.2 | 7.7 | 7.6 | 7.7 | 11.49 | 0.527 | 7.8 / 75.6 / 0.700 |
| grec_testB | gref_only | raw | 88.7 | 74.0 | 93.9 | 9.3 | 9.0 | 9.3 | 12.47 | 0.565 | - |
| grec_testB | gref_only | temp | 88.7 | 74.0 | 93.9 | 9.2 | 9.0 | 9.2 | 12.46 | 0.563 | 8.7 / 76.7 / 0.601 |
| grec_testB | gref_only | platt | 88.7 | 74.9 | 93.2 | 9.0 | 8.7 | 9.0 | 12.40 | 0.554 | 8.8 / 77.8 / 0.588 |
| grec_testB | gref_only | beta | 88.7 | 74.2 | 93.8 | 9.4 | 9.1 | 9.4 | 12.46 | 0.534 | 8.6 / 76.9 / 0.592 |
| grec_testB | gref_only | iso | 88.7 | 73.4 | 94.3 | 10.0 | 9.3 | 10.0 | 12.62 | 0.537 | 9.3 / 77.1 / 0.596 |
| grec_testB | gref_only | multi | 88.5 | 74.9 | 92.9 | 7.1 | 7.0 | 7.1 | 12.17 | 0.475 | 7.5 / 79.0 / 0.565 |
| calib_natural | all | raw | 99.6 | 95.7 | 99.0 | 0.6 | 0.5 | 0.6 | 1.52 | 0.059 | - |
| calib_natural | all | temp | 99.6 | 95.7 | 99.0 | 0.6 | 0.5 | 0.6 | 1.52 | 0.059 | 0.6 / 95.6 / 0.727 |
| calib_natural | all | platt | 99.6 | 96.1 | 98.8 | 0.5 | 0.5 | 0.4 | 1.51 | 0.059 | 0.5 / 96.1 / 0.723 |
| calib_natural | all | beta | 99.6 | 95.7 | 99.0 | 0.3 | 0.3 | 0.4 | 1.51 | 0.059 | 0.3 / 95.7 / 0.723 |
| calib_natural | all | iso | 99.6 | 95.4 | 99.1 | 0.3 | 0.1 | 0.1 | 1.49 | 0.058 | 0.3 / 95.4 / 0.723 |
| calib_natural | all | multi | 99.6 | 96.0 | 98.8 | 1.3 | 1.3 | 1.3 | 1.57 | 0.062 | 1.3 / 96.1 / 0.720 |
| grefcalib | all | raw | 99.5 | 96.7 | 98.7 | 1.7 | 1.8 | 1.7 | 1.18 | 0.046 | - |
| grefcalib | all | temp | 99.5 | 96.7 | 98.7 | 1.7 | 1.8 | 1.7 | 1.18 | 0.046 | 0.4 / 94.7 / 0.894 |
| grefcalib | all | platt | 99.5 | 97.3 | 98.4 | 2.0 | 2.1 | 2.0 | 1.30 | 0.050 | 0.5 / 94.9 / 0.892 |
| grefcalib | all | beta | 99.5 | 96.9 | 98.6 | 2.0 | 2.1 | 2.0 | 1.23 | 0.050 | 0.5 / 94.7 / 0.891 |
| grefcalib | all | iso | 99.5 | 96.5 | 98.8 | 1.9 | 1.9 | 1.9 | 1.25 | 0.049 | 0.6 / 95.6 / 0.892 |
| grefcalib | all | multi | 99.4 | 97.3 | 98.4 | 2.8 | 2.8 | 2.8 | 1.38 | 0.056 | 1.1 / 94.2 / 0.892 |
| guisyn_test | natural_fit | raw | 86.7 | 75.6 | 86.5 | 11.5 | 11.5 | 11.5 | 12.79 | 0.511 | - |
| guisyn_test | natural_fit | temp | 86.7 | 75.6 | 86.5 | 11.5 | 11.5 | 11.5 | 12.78 | 0.509 | 12.4 / 76.4 / 0.683 |
| guisyn_test | natural_fit | platt | 86.7 | 76.5 | 85.2 | 12.6 | 12.5 | 12.5 | 13.36 | 0.528 | 14.0 / 77.8 / 0.664 |
| guisyn_test | natural_fit | beta | 86.7 | 75.7 | 86.3 | 12.0 | 11.9 | 11.9 | 13.14 | 0.528 | 13.1 / 76.8 / 0.672 |
| guisyn_test | natural_fit | iso | 86.7 | 74.9 | 87.3 | 12.0 | 12.0 | 12.1 | 13.40 | 0.523 | 12.8 / 76.0 / 0.676 |
| guisyn_test | natural_fit | multi | 85.5 | 73.8 | 86.8 | 11.0 | 10.9 | 10.8 | 12.65 | 0.461 | 12.4 / 75.7 / 0.675 |
| guisyn_test | gui_fit | raw | 86.7 | 75.6 | 86.5 | 11.5 | 11.5 | 11.5 | 12.79 | 0.511 | - |
| guisyn_test | gui_fit | temp | 86.7 | 75.6 | 86.5 | 10.3 | 10.7 | 10.6 | 12.31 | 0.396 | 25.5 / 85.7 / 0.493 |
| guisyn_test | gui_fit | platt | 86.7 | 60.2 | 95.4 | 5.8 | 5.9 | 5.6 | 10.80 | 0.365 | 6.1 / 64.4 / 0.732 |
| guisyn_test | gui_fit | beta | 86.7 | 57.8 | 96.0 | 1.8 | 3.3 | 2.3 | 10.49 | 0.351 | 2.2 / 60.2 / 0.734 |
| guisyn_test | gui_fit | iso | 86.7 | 63.4 | 94.4 | 1.6 | 2.4 | 2.3 | 10.42 | 0.345 | 2.9 / 65.3 / 0.721 |
| guisyn_test | gui_fit | multi | 96.2 | 81.5 | 95.7 | 1.3 | 1.4 | 1.3 | 6.09 | 0.211 | 1.3 / 81.6 / 0.760 |
| venus | natural_fit | raw | 77.7 | 83.8 | 57.8 | 35.1 | 35.2 | 34.8 | 30.11 | 1.152 | - |
| venus | natural_fit | temp | 77.7 | 83.8 | 57.8 | 35.1 | 35.2 | 34.8 | 30.09 | 1.148 | 50.5 / 90.8 / 0.351 |
| venus | natural_fit | platt | 77.7 | 85.0 | 55.3 | 37.3 | 37.4 | 37.0 | 31.98 | 1.242 | 53.8 / 92.4 / 0.318 |
| venus | natural_fit | beta | 77.7 | 83.8 | 57.3 | 36.3 | 36.3 | 35.9 | 31.46 | 1.361 | 52.2 / 91.1 / 0.333 |
| venus | natural_fit | iso | 77.7 | 83.3 | 59.2 | 36.3 | 36.5 | 36.2 | 32.25 | 1.274 | 50.4 / 90.4 / 0.351 |
| venus | natural_fit | multi | 78.5 | 84.0 | 59.3 | 33.6 | 33.6 | 33.3 | 27.36 | 0.905 | 54.9 / 93.9 / 0.305 |
| venus | gui_fit | raw | 77.7 | 83.8 | 57.8 | 35.1 | 35.2 | 34.8 | 30.11 | 1.152 | - |
| venus | gui_fit | temp | 77.7 | 83.8 | 57.8 | 35.0 | 35.0 | 35.0 | 25.55 | 0.741 | 85.3 / 100.0 / 0.001 |
| venus | gui_fit | platt | 77.7 | 63.8 | 78.5 | 21.5 | 21.5 | 21.5 | 16.83 | 0.509 | 61.5 / 97.3 / 0.240 |
| venus | gui_fit | beta | 77.7 | 60.7 | 80.5 | 19.9 | 19.9 | 19.9 | 17.10 | 0.549 | 49.4 / 90.3 / 0.361 |
| venus | gui_fit | iso | 77.7 | 67.3 | 76.0 | 20.8 | 20.8 | 20.8 | 17.30 | 0.558 | 55.5 / 95.4 / 0.299 |
| venus | gui_fit | multi | 58.3 | 36.3 | 71.5 | 23.8 | 23.7 | 23.7 | 24.56 | 0.750 | 28.3 / 46.3 / 0.605 |

## κ̂ calibration (natural: event IoU>=0.5; GUI: click in box)

| set | fit | method | AUROC | AURC | ECE15 | ECE15-mass | Brier | mean κ̂ | rate |
|---|---|---|---|---|---|---|---|---|---|
| refcoco_val | natural_fit | raw | 90.1 | 0.80 | 0.8 | 0.7 | 3.69 | 95.5 | 94.9 |
| refcoco_val | natural_fit | temp | 90.1 | 0.80 | 0.5 | 0.5 | 3.68 | 95.2 | 94.9 |
| refcoco_val | natural_fit | platt | 90.1 | 0.80 | 0.6 | 0.4 | 3.68 | 94.9 | 94.9 |
| refcoco_val | natural_fit | beta | 90.1 | 0.80 | 0.5 | 0.4 | 3.67 | 94.9 | 94.9 |
| refcoco_val | natural_fit | iso | 90.1 | 0.80 | 0.6 | 0.5 | 3.69 | 94.8 | 94.9 |
| refcoco_val | natural_fit | multi | 90.1 | 0.79 | 0.6 | 0.5 | 3.67 | 94.5 | 94.9 |
| refcoco_testA | natural_fit | raw | 91.8 | 0.54 | 0.8 | 1.0 | 2.83 | 96.7 | 96.1 |
| refcoco_testA | natural_fit | temp | 91.8 | 0.54 | 0.5 | 0.9 | 2.82 | 96.5 | 96.1 |
| refcoco_testA | natural_fit | platt | 91.8 | 0.54 | 0.5 | 0.7 | 2.81 | 96.3 | 96.1 |
| refcoco_testA | natural_fit | beta | 91.8 | 0.54 | 0.5 | 0.6 | 2.80 | 96.3 | 96.1 |
| refcoco_testA | natural_fit | iso | 91.8 | 0.63 | 0.7 | 0.5 | 2.80 | 96.3 | 96.1 |
| refcoco_testA | natural_fit | multi | 91.8 | 0.54 | 0.6 | 0.8 | 2.82 | 95.9 | 96.1 |
| refcoco_testB | natural_fit | raw | 90.5 | 1.14 | 1.1 | 1.3 | 4.87 | 93.7 | 92.8 |
| refcoco_testB | natural_fit | temp | 90.5 | 1.14 | 0.9 | 1.1 | 4.86 | 93.3 | 92.8 |
| refcoco_testB | natural_fit | platt | 90.5 | 1.14 | 0.6 | 0.7 | 4.84 | 92.9 | 92.8 |
| refcoco_testB | natural_fit | beta | 90.5 | 1.14 | 0.6 | 0.5 | 4.84 | 92.9 | 92.8 |
| refcoco_testB | natural_fit | iso | 90.5 | 1.14 | 0.8 | 0.6 | 4.84 | 92.7 | 92.8 |
| refcoco_testB | natural_fit | multi | 90.6 | 1.11 | 1.3 | 0.9 | 4.82 | 92.4 | 92.8 |
| refcocoplus_val | natural_fit | raw | 91.1 | 1.22 | 0.8 | 0.7 | 5.10 | 92.4 | 91.9 |
| refcocoplus_val | natural_fit | temp | 91.1 | 1.22 | 0.5 | 0.6 | 5.10 | 92.1 | 91.9 |
| refcocoplus_val | natural_fit | platt | 91.1 | 1.22 | 0.9 | 0.6 | 5.10 | 91.6 | 91.9 |
| refcocoplus_val | natural_fit | beta | 91.1 | 1.22 | 0.8 | 0.7 | 5.11 | 91.6 | 91.9 |
| refcocoplus_val | natural_fit | iso | 91.1 | 1.22 | 0.7 | 0.8 | 5.11 | 91.4 | 91.9 |
| refcocoplus_val | natural_fit | multi | 91.1 | 1.22 | 0.9 | 0.8 | 5.13 | 91.2 | 91.9 |
| refcocoplus_testA | natural_fit | raw | 91.8 | 0.66 | 0.5 | 0.8 | 3.55 | 95.3 | 94.9 |
| refcocoplus_testA | natural_fit | temp | 91.8 | 0.66 | 0.3 | 0.8 | 3.54 | 95.0 | 94.9 |
| refcocoplus_testA | natural_fit | platt | 91.8 | 0.66 | 0.7 | 0.8 | 3.54 | 94.7 | 94.9 |
| refcocoplus_testA | natural_fit | beta | 91.8 | 0.66 | 0.6 | 0.8 | 3.54 | 94.7 | 94.9 |
| refcocoplus_testA | natural_fit | iso | 91.8 | 0.66 | 0.6 | 0.6 | 3.55 | 94.7 | 94.9 |
| refcocoplus_testA | natural_fit | multi | 91.8 | 0.66 | 0.9 | 0.9 | 3.56 | 94.3 | 94.9 |
| refcocoplus_testB | natural_fit | raw | 89.8 | 2.29 | 1.7 | 1.6 | 7.28 | 88.9 | 87.7 |
| refcocoplus_testB | natural_fit | temp | 89.8 | 2.29 | 1.3 | 1.4 | 7.28 | 88.5 | 87.7 |
| refcocoplus_testB | natural_fit | platt | 89.8 | 2.29 | 1.4 | 1.2 | 7.26 | 87.7 | 87.7 |
| refcocoplus_testB | natural_fit | beta | 89.8 | 2.29 | 1.4 | 1.4 | 7.28 | 87.7 | 87.7 |
| refcocoplus_testB | natural_fit | iso | 89.8 | 2.29 | 1.5 | 1.2 | 7.26 | 87.4 | 87.7 |
| refcocoplus_testB | natural_fit | multi | 89.8 | 2.31 | 1.3 | 1.5 | 7.30 | 87.5 | 87.7 |
| refcocog_val | natural_fit | raw | 87.7 | 1.57 | 1.5 | 1.5 | 5.56 | 93.6 | 92.2 |
| refcocog_val | natural_fit | temp | 87.7 | 1.57 | 1.2 | 1.1 | 5.54 | 93.2 | 92.2 |
| refcocog_val | natural_fit | platt | 87.7 | 1.57 | 1.2 | 1.1 | 5.53 | 92.8 | 92.2 |
| refcocog_val | natural_fit | beta | 87.7 | 1.57 | 0.9 | 0.9 | 5.52 | 92.7 | 92.2 |
| refcocog_val | natural_fit | iso | 87.7 | 1.57 | 1.0 | 0.9 | 5.53 | 92.5 | 92.2 |
| refcocog_val | natural_fit | multi | 87.7 | 1.59 | 0.8 | 0.8 | 5.53 | 92.5 | 92.2 |
| refcocog_test | natural_fit | raw | 89.0 | 1.32 | 1.1 | 1.1 | 5.21 | 93.4 | 92.5 |
| refcocog_test | natural_fit | temp | 89.0 | 1.32 | 0.7 | 0.8 | 5.20 | 93.0 | 92.5 |
| refcocog_test | natural_fit | platt | 89.0 | 1.32 | 0.7 | 0.9 | 5.20 | 92.6 | 92.5 |
| refcocog_test | natural_fit | beta | 89.0 | 1.32 | 0.8 | 0.8 | 5.21 | 92.5 | 92.5 |
| refcocog_test | natural_fit | iso | 89.0 | 1.32 | 0.7 | 0.7 | 5.22 | 92.3 | 92.5 |
| refcocog_test | natural_fit | multi | 88.9 | 1.33 | 0.5 | 0.6 | 5.22 | 92.3 | 92.5 |
| grec_val | natural_fit | raw | 90.2 | 0.79 | 0.9 | 0.9 | 3.70 | 95.5 | 94.8 |
| grec_val | natural_fit | temp | 90.2 | 0.79 | 0.5 | 0.7 | 3.69 | 95.2 | 94.8 |
| grec_val | natural_fit | platt | 90.2 | 0.79 | 0.5 | 0.5 | 3.69 | 94.9 | 94.8 |
| grec_val | natural_fit | beta | 90.2 | 0.79 | 0.5 | 0.5 | 3.68 | 94.9 | 94.8 |
| grec_val | natural_fit | iso | 90.2 | 0.79 | 0.5 | 0.5 | 3.70 | 94.8 | 94.8 |
| grec_val | natural_fit | multi | 90.2 | 0.79 | 0.5 | 0.6 | 3.68 | 94.5 | 94.8 |
| grec_testA | natural_fit | raw | 94.0 | 0.66 | 1.6 | 1.9 | 3.51 | 95.7 | 94.2 |
| grec_testA | natural_fit | temp | 94.0 | 0.66 | 1.4 | 1.9 | 3.49 | 95.5 | 94.2 |
| grec_testA | natural_fit | platt | 94.0 | 0.66 | 1.3 | 1.6 | 3.42 | 95.2 | 94.2 |
| grec_testA | natural_fit | beta | 94.0 | 0.66 | 1.4 | 1.5 | 3.42 | 95.3 | 94.2 |
| grec_testA | natural_fit | iso | 94.0 | 0.75 | 1.2 | 1.3 | 3.42 | 95.2 | 94.2 |
| grec_testA | natural_fit | multi | 93.9 | 0.66 | 1.5 | 1.6 | 3.49 | 94.9 | 94.2 |
| grec_testB | natural_fit | raw | 93.4 | 2.12 | 3.7 | 4.1 | 7.00 | 89.8 | 86.0 |
| grec_testB | natural_fit | temp | 93.4 | 2.12 | 3.5 | 3.9 | 6.98 | 89.4 | 86.0 |
| grec_testB | natural_fit | platt | 93.4 | 2.12 | 2.8 | 3.1 | 6.76 | 88.6 | 86.0 |
| grec_testB | natural_fit | beta | 93.4 | 2.12 | 2.7 | 3.1 | 6.81 | 88.7 | 86.0 |
| grec_testB | natural_fit | iso | 93.4 | 2.12 | 2.9 | 3.0 | 6.80 | 88.5 | 86.0 |
| grec_testB | natural_fit | multi | 93.5 | 2.10 | 3.2 | 3.3 | 6.76 | 88.3 | 86.0 |
| calib_natural | natural_fit | raw | 91.8 | 0.88 | 0.9 | 1.1 | 4.53 | 93.9 | 93.2 |
| calib_natural | natural_fit | temp | 91.8 | 0.88 | 0.6 | 0.9 | 4.52 | 93.6 | 93.2 |
| calib_natural | natural_fit | platt | 91.8 | 0.88 | 0.6 | 0.8 | 4.51 | 93.2 | 93.2 |
| calib_natural | natural_fit | beta | 91.8 | 0.88 | 0.7 | 0.8 | 4.52 | 93.2 | 93.2 |
| calib_natural | natural_fit | iso | 91.8 | 0.88 | 0.4 | 0.4 | 4.49 | 93.0 | 93.2 |
| calib_natural | natural_fit | multi | 91.9 | 0.87 | 0.8 | 0.7 | 4.50 | 92.8 | 93.2 |
| guisyn_test | natural_fit | raw | 90.8 | 3.81 | 15.9 | 15.9 | 11.88 | 66.0 | 81.9 |
| guisyn_test | natural_fit | temp | 90.8 | 3.81 | 16.2 | 16.2 | 11.93 | 65.6 | 81.9 |
| guisyn_test | natural_fit | platt | 90.8 | 3.81 | 18.4 | 18.4 | 13.01 | 63.5 | 81.9 |
| guisyn_test | natural_fit | beta | 90.8 | 3.81 | 17.5 | 17.5 | 12.58 | 64.4 | 81.9 |
| guisyn_test | natural_fit | iso | 90.3 | 4.10 | 17.8 | 17.8 | 12.97 | 64.0 | 81.9 |
| guisyn_test | natural_fit | multi | 90.4 | 3.93 | 63.3 | 63.3 | 52.36 | 18.5 | 81.9 |
| guisyn_test | gui_fit | raw | 90.8 | 3.81 | 15.9 | 15.9 | 11.88 | 66.0 | 81.9 |
| guisyn_test | gui_fit | temp | 90.8 | 3.81 | 14.0 | 14.0 | 11.81 | 67.9 | 81.9 |
| guisyn_test | gui_fit | platt | 90.8 | 3.81 | 1.0 | 1.0 | 8.51 | 82.3 | 81.9 |
| guisyn_test | gui_fit | beta | 90.8 | 3.81 | 0.8 | 0.7 | 8.50 | 82.3 | 81.9 |
| guisyn_test | gui_fit | iso | 90.8 | 3.81 | 0.9 | 0.8 | 8.51 | 82.4 | 81.9 |
| guisyn_test | gui_fit | multi | 91.0 | 3.79 | 0.7 | 0.8 | 8.38 | 81.9 | 81.9 |
| ss2 | natural_fit | raw | 80.6 | 2.99 | 28.2 | 28.2 | 18.22 | 61.9 | 90.1 |
| ss2 | natural_fit | temp | 80.6 | 2.99 | 28.4 | 28.4 | 18.17 | 61.6 | 90.1 |
| ss2 | natural_fit | platt | 80.6 | 2.99 | 31.3 | 31.3 | 20.45 | 58.8 | 90.1 |
| ss2 | natural_fit | beta | 80.6 | 2.99 | 30.4 | 30.4 | 19.23 | 59.7 | 90.1 |
| ss2 | natural_fit | iso | 80.2 | 3.12 | 31.4 | 31.4 | 20.59 | 58.7 | 90.1 |
| ss2 | natural_fit | multi | 82.5 | 2.78 | 74.8 | 74.8 | 65.07 | 15.3 | 90.1 |
| ss2 | gui_fit | raw | 80.6 | 2.99 | 28.2 | 28.2 | 18.22 | 61.9 | 90.1 |
| ss2 | gui_fit | temp | 80.6 | 2.99 | 26.7 | 26.7 | 18.81 | 63.4 | 90.1 |
| ss2 | gui_fit | platt | 80.6 | 2.99 | 7.1 | 7.3 | 8.87 | 83.0 | 90.1 |
| ss2 | gui_fit | beta | 80.6 | 2.99 | 7.4 | 7.6 | 8.95 | 82.7 | 90.1 |
| ss2 | gui_fit | iso | 80.6 | 2.99 | 7.2 | 7.5 | 8.86 | 82.9 | 90.1 |
| ss2 | gui_fit | multi | 81.6 | 2.88 | 7.2 | 7.3 | 8.42 | 83.0 | 90.1 |
| venus | natural_fit | raw | 81.5 | 26.24 | 14.3 | 14.3 | 20.14 | 36.6 | 50.9 |
| venus | natural_fit | temp | 81.5 | 26.24 | 14.0 | 14.0 | 20.01 | 36.9 | 50.9 |
| venus | natural_fit | platt | 81.5 | 26.24 | 17.2 | 17.2 | 21.24 | 33.7 | 50.9 |
| venus | natural_fit | beta | 81.5 | 26.24 | 14.1 | 14.0 | 20.22 | 37.0 | 50.9 |
| venus | natural_fit | iso | 79.0 | 28.93 | 13.7 | 14.4 | 21.16 | 37.3 | 50.9 |
| venus | natural_fit | multi | 81.3 | 26.27 | 44.9 | 44.9 | 42.36 | 6.0 | 50.9 |
| venus | gui_fit | raw | 81.5 | 26.24 | 14.3 | 14.3 | 20.14 | 36.6 | 50.9 |
| venus | gui_fit | temp | 81.5 | 26.24 | 16.0 | 16.2 | 21.12 | 35.2 | 50.9 |
| venus | gui_fit | platt | 81.5 | 26.24 | 6.7 | 6.5 | 18.06 | 57.2 | 50.9 |
| venus | gui_fit | beta | 81.5 | 26.24 | 6.5 | 6.4 | 18.02 | 57.2 | 50.9 |
| venus | gui_fit | iso | 81.5 | 26.24 | 6.7 | 6.6 | 18.02 | 57.1 | 50.9 |
| venus | gui_fit | multi | 81.8 | 26.24 | 6.7 | 6.7 | 17.97 | 57.3 | 50.9 |

## Box confidence ĝ·κ̂ vs (present ∧ IoU>=0.5)  [the v0→v1 regression metric]

| set | method | ECE15 | ECE15-mass | Brier | π̂ (EM) |
|---|---|---|---|---|---|
| refcoco_val | raw | 0.92 | 1.04 | 4.11 | - |
| refcoco_val | evalpy_T | 0.93 | 1.05 | 4.11 | - |
| refcoco_val | g:platt|k:platt | 1.31 | 1.53 | 4.19 | - |
| refcoco_val | g:platt+em|k:platt | 0.54 | 0.43 | 3.69 | 0.999 |
| refcoco_val | g:platt|k:iso | 1.61 | 1.80 | 4.20 | - |
| refcoco_val | g:platt+em|k:iso | 0.56 | 0.46 | 3.70 | 0.999 |
| refcoco_val | g:platt|k:multi | 1.74 | 1.80 | 4.20 | - |
| refcoco_val | g:platt+em|k:multi | 0.62 | 0.55 | 3.68 | 0.999 |
| refcoco_val | g:iso|k:platt | 1.26 | 1.45 | 4.15 | - |
| refcoco_val | g:iso+em|k:platt | 0.54 | 0.43 | 3.69 | 0.999 |
| refcoco_val | g:iso|k:iso | 1.53 | 1.67 | 4.17 | - |
| refcoco_val | g:iso+em|k:iso | 0.57 | 0.46 | 3.70 | 0.999 |
| refcoco_val | g:iso|k:multi | 1.69 | 1.69 | 4.16 | - |
| refcoco_val | g:iso+em|k:multi | 0.63 | 0.56 | 3.68 | 0.999 |
| refcoco_val | kappa_only_raw | 0.82 | 0.74 | 3.69 | - |
| refcoco_testA | raw | 1.06 | 1.06 | 3.29 | - |
| refcoco_testA | evalpy_T | 1.05 | 1.06 | 3.29 | - |
| refcoco_testA | g:platt|k:platt | 1.06 | 1.34 | 3.33 | - |
| refcoco_testA | g:platt+em|k:platt | 0.52 | 0.66 | 2.83 | 0.999 |
| refcoco_testA | g:platt|k:iso | 1.09 | 1.35 | 3.33 | - |
| refcoco_testA | g:platt+em|k:iso | 0.67 | 0.46 | 2.83 | 0.999 |
| refcoco_testA | g:platt|k:multi | 1.47 | 1.60 | 3.35 | - |
| refcoco_testA | g:platt+em|k:multi | 0.65 | 0.71 | 2.85 | 0.999 |
| refcoco_testA | g:iso|k:platt | 1.19 | 1.35 | 3.32 | - |
| refcoco_testA | g:iso+em|k:platt | 0.49 | 0.65 | 2.83 | 0.999 |
| refcoco_testA | g:iso|k:iso | 1.15 | 1.15 | 3.32 | - |
| refcoco_testA | g:iso+em|k:iso | 0.66 | 0.46 | 2.83 | 0.999 |
| refcoco_testA | g:iso|k:multi | 1.64 | 1.52 | 3.34 | - |
| refcoco_testA | g:iso+em|k:multi | 0.62 | 0.71 | 2.85 | 0.999 |
| refcoco_testA | kappa_only_raw | 0.85 | 1.00 | 2.83 | - |
| refcoco_testB | raw | 1.04 | 1.43 | 5.30 | - |
| refcoco_testB | evalpy_T | 1.05 | 1.43 | 5.30 | - |
| refcoco_testB | g:platt|k:platt | 1.25 | 1.65 | 5.38 | - |
| refcoco_testB | g:platt+em|k:platt | 0.61 | 0.75 | 4.83 | 0.999 |
| refcoco_testB | g:platt|k:iso | 1.80 | 1.78 | 5.39 | - |
| refcoco_testB | g:platt+em|k:iso | 0.85 | 0.59 | 4.84 | 0.999 |
| refcoco_testB | g:platt|k:multi | 1.77 | 1.77 | 5.37 | - |
| refcoco_testB | g:platt+em|k:multi | 1.23 | 0.91 | 4.82 | 0.999 |
| refcoco_testB | g:iso|k:platt | 1.34 | 1.59 | 5.34 | - |
| refcoco_testB | g:iso+em|k:platt | 0.63 | 0.75 | 4.83 | 0.999 |
| refcoco_testB | g:iso|k:iso | 1.60 | 1.63 | 5.35 | - |
| refcoco_testB | g:iso+em|k:iso | 0.85 | 0.56 | 4.84 | 0.999 |
| refcoco_testB | g:iso|k:multi | 1.74 | 1.72 | 5.34 | - |
| refcoco_testB | g:iso+em|k:multi | 1.21 | 0.91 | 4.82 | 0.999 |
| refcoco_testB | kappa_only_raw | 1.07 | 1.26 | 4.87 | - |
| refcocoplus_val | raw | 1.24 | 1.48 | 5.60 | - |
| refcocoplus_val | evalpy_T | 1.24 | 1.49 | 5.60 | - |
| refcocoplus_val | g:platt|k:platt | 2.25 | 2.38 | 5.72 | - |
| refcocoplus_val | g:platt+em|k:platt | 0.98 | 0.68 | 5.11 | 0.999 |
| refcocoplus_val | g:platt|k:iso | 2.39 | 2.61 | 5.74 | - |
| refcocoplus_val | g:platt+em|k:iso | 0.75 | 0.79 | 5.12 | 0.999 |
| refcocoplus_val | g:platt|k:multi | 2.64 | 2.65 | 5.76 | - |
| refcocoplus_val | g:platt+em|k:multi | 0.86 | 0.84 | 5.14 | 0.999 |
| refcocoplus_val | g:iso|k:platt | 2.14 | 2.23 | 5.68 | - |
| refcocoplus_val | g:iso+em|k:platt | 0.99 | 0.68 | 5.11 | 0.999 |
| refcocoplus_val | g:iso|k:iso | 2.38 | 2.42 | 5.69 | - |
| refcocoplus_val | g:iso+em|k:iso | 0.77 | 0.82 | 5.12 | 0.999 |
| refcocoplus_val | g:iso|k:multi | 2.53 | 2.53 | 5.72 | - |
| refcocoplus_val | g:iso+em|k:multi | 0.89 | 0.83 | 5.14 | 0.999 |
| refcocoplus_val | kappa_only_raw | 0.80 | 0.69 | 5.10 | - |
| refcocoplus_testA | raw | 1.23 | 1.62 | 3.99 | - |
| refcocoplus_testA | evalpy_T | 1.22 | 1.62 | 4.00 | - |
| refcocoplus_testA | g:platt|k:platt | 1.74 | 2.18 | 4.10 | - |
| refcocoplus_testA | g:platt+em|k:platt | 0.64 | 0.76 | 3.54 | 0.999 |
| refcocoplus_testA | g:platt|k:iso | 1.77 | 2.09 | 4.10 | - |
| refcocoplus_testA | g:platt+em|k:iso | 0.57 | 0.66 | 3.55 | 0.999 |
| refcocoplus_testA | g:platt|k:multi | 2.10 | 2.29 | 4.12 | - |
| refcocoplus_testA | g:platt+em|k:multi | 0.90 | 0.85 | 3.56 | 0.999 |
| refcocoplus_testA | g:iso|k:platt | 1.70 | 2.10 | 4.04 | - |
| refcocoplus_testA | g:iso+em|k:platt | 0.63 | 0.77 | 3.54 | 0.999 |
| refcocoplus_testA | g:iso|k:iso | 1.77 | 1.90 | 4.04 | - |
| refcocoplus_testA | g:iso+em|k:iso | 0.59 | 0.67 | 3.55 | 0.999 |
| refcocoplus_testA | g:iso|k:multi | 2.03 | 2.22 | 4.06 | - |
| refcocoplus_testA | g:iso+em|k:multi | 0.91 | 0.85 | 3.56 | 0.999 |
| refcocoplus_testA | kappa_only_raw | 0.52 | 0.84 | 3.55 | - |
| refcocoplus_testB | raw | 1.64 | 1.72 | 7.75 | - |
| refcocoplus_testB | evalpy_T | 1.66 | 1.72 | 7.75 | - |
| refcocoplus_testB | g:platt|k:platt | 2.10 | 2.33 | 7.86 | - |
| refcocoplus_testB | g:platt+em|k:platt | 1.44 | 1.31 | 7.25 | 0.999 |
| refcocoplus_testB | g:platt|k:iso | 2.66 | 2.21 | 7.87 | - |
| refcocoplus_testB | g:platt+em|k:iso | 1.50 | 1.16 | 7.26 | 0.999 |
| refcocoplus_testB | g:platt|k:multi | 2.11 | 2.19 | 7.90 | - |
| refcocoplus_testB | g:platt+em|k:multi | 1.20 | 1.55 | 7.29 | 0.999 |
| refcocoplus_testB | g:iso|k:platt | 2.33 | 2.24 | 7.82 | - |
| refcocoplus_testB | g:iso+em|k:platt | 1.49 | 1.31 | 7.25 | 0.999 |
| refcocoplus_testB | g:iso|k:iso | 2.81 | 2.23 | 7.83 | - |
| refcocoplus_testB | g:iso+em|k:iso | 1.53 | 1.20 | 7.26 | 0.999 |
| refcocoplus_testB | g:iso|k:multi | 2.28 | 2.06 | 7.85 | - |
| refcocoplus_testB | g:iso+em|k:multi | 1.23 | 1.54 | 7.29 | 0.999 |
| refcocoplus_testB | kappa_only_raw | 1.69 | 1.55 | 7.28 | - |
| refcocog_val | raw | 2.28 | 2.19 | 6.30 | - |
| refcocog_val | evalpy_T | 2.29 | 2.19 | 6.30 | - |
| refcocog_val | g:platt|k:platt | 2.58 | 2.74 | 6.45 | - |
| refcocog_val | g:platt+em|k:platt | 1.16 | 1.03 | 5.53 | 0.999 |
| refcocog_val | g:platt|k:iso | 2.52 | 3.03 | 6.47 | - |
| refcocog_val | g:platt+em|k:iso | 0.97 | 0.84 | 5.53 | 0.999 |
| refcocog_val | g:platt|k:multi | 2.45 | 2.64 | 6.46 | - |
| refcocog_val | g:platt+em|k:multi | 0.76 | 0.71 | 5.52 | 0.999 |
| refcocog_val | g:iso|k:platt | 2.21 | 2.57 | 6.40 | - |
| refcocog_val | g:iso+em|k:platt | 1.17 | 1.03 | 5.53 | 0.999 |
| refcocog_val | g:iso|k:iso | 2.25 | 2.72 | 6.42 | - |
| refcocog_val | g:iso+em|k:iso | 0.95 | 0.81 | 5.54 | 0.999 |
| refcocog_val | g:iso|k:multi | 2.36 | 2.44 | 6.41 | - |
| refcocog_val | g:iso+em|k:multi | 0.74 | 0.71 | 5.52 | 0.999 |
| refcocog_val | kappa_only_raw | 1.52 | 1.46 | 5.56 | - |
| refcocog_test | raw | 1.81 | 1.96 | 5.85 | - |
| refcocog_test | evalpy_T | 1.81 | 1.95 | 5.85 | - |
| refcocog_test | g:platt|k:platt | 2.60 | 2.83 | 6.02 | - |
| refcocog_test | g:platt+em|k:platt | 0.66 | 0.85 | 5.21 | 0.999 |
| refcocog_test | g:platt|k:iso | 2.84 | 3.11 | 6.05 | - |
| refcocog_test | g:platt+em|k:iso | 0.62 | 0.74 | 5.22 | 0.999 |
| refcocog_test | g:platt|k:multi | 2.89 | 2.96 | 6.04 | - |
| refcocog_test | g:platt+em|k:multi | 0.47 | 0.62 | 5.22 | 0.999 |
| refcocog_test | g:iso|k:platt | 2.39 | 2.64 | 5.95 | - |
| refcocog_test | g:iso+em|k:platt | 0.67 | 0.85 | 5.21 | 0.999 |
| refcocog_test | g:iso|k:iso | 2.64 | 2.88 | 5.98 | - |
| refcocog_test | g:iso+em|k:iso | 0.64 | 0.74 | 5.22 | 0.999 |
| refcocog_test | g:iso|k:multi | 2.69 | 2.70 | 5.97 | - |
| refcocog_test | g:iso+em|k:multi | 0.49 | 0.62 | 5.22 | 0.999 |
| refcocog_test | kappa_only_raw | 1.12 | 1.08 | 5.21 | - |
| grec_val | raw | 5.72 | 5.73 | 6.12 | - |
| grec_val | evalpy_T | 5.72 | 5.73 | 6.12 | - |
| grec_val | g:platt|k:platt | 4.83 | 4.84 | 5.82 | - |
| grec_val | g:platt+em|k:platt | 3.65 | 3.66 | 5.38 | 0.592 |
| grec_val | g:platt|k:iso | 4.74 | 4.75 | 5.80 | - |
| grec_val | g:platt+em|k:iso | 3.56 | 3.57 | 5.36 | 0.592 |
| grec_val | g:platt|k:multi | 4.58 | 4.59 | 5.79 | - |
| grec_val | g:platt+em|k:multi | 3.40 | 3.41 | 5.35 | 0.592 |
| grec_val | g:iso|k:platt | 4.86 | 4.86 | 5.97 | - |
| grec_val | g:iso+em|k:platt | 3.78 | 3.79 | 5.54 | 0.594 |
| grec_val | g:iso|k:iso | 4.76 | 4.77 | 5.94 | - |
| grec_val | g:iso+em|k:iso | 3.69 | 3.70 | 5.52 | 0.594 |
| grec_val | g:iso|k:multi | 4.61 | 4.62 | 5.94 | - |
| grec_val | g:iso+em|k:multi | 3.54 | 3.56 | 5.51 | 0.594 |
| grec_val | kappa_only_raw | 31.14 | 31.14 | 25.62 | - |
| grec_testA | raw | 8.11 | 8.22 | 8.32 | - |
| grec_testA | evalpy_T | 8.10 | 8.22 | 8.32 | - |
| grec_testA | g:platt|k:platt | 7.41 | 7.49 | 7.97 | - |
| grec_testA | g:platt+em|k:platt | 6.67 | 6.72 | 7.56 | 0.629 |
| grec_testA | g:platt|k:iso | 7.36 | 7.45 | 7.92 | - |
| grec_testA | g:platt+em|k:iso | 6.62 | 6.68 | 7.52 | 0.629 |
| grec_testA | g:platt|k:multi | 7.17 | 7.26 | 7.93 | - |
| grec_testA | g:platt+em|k:multi | 6.43 | 6.49 | 7.53 | 0.629 |
| grec_testA | g:iso|k:platt | 7.51 | 7.55 | 8.16 | - |
| grec_testA | g:iso+em|k:platt | 6.84 | 6.85 | 7.79 | 0.631 |
| grec_testA | g:iso|k:iso | 7.44 | 7.51 | 8.12 | - |
| grec_testA | g:iso+em|k:iso | 6.80 | 6.82 | 7.74 | 0.631 |
| grec_testA | g:iso|k:multi | 7.26 | 7.32 | 8.12 | - |
| grec_testA | g:iso+em|k:multi | 6.59 | 6.63 | 7.75 | 0.631 |
| grec_testA | kappa_only_raw | 30.32 | 30.32 | 25.21 | - |
| grec_testB | raw | 11.43 | 11.44 | 12.16 | - |
| grec_testB | evalpy_T | 11.43 | 11.44 | 12.15 | - |
| grec_testB | g:platt|k:platt | 10.50 | 10.54 | 11.72 | - |
| grec_testB | g:platt+em|k:platt | 9.35 | 9.42 | 11.11 | 0.587 |
| grec_testB | g:platt|k:iso | 10.34 | 10.38 | 11.66 | - |
| grec_testB | g:platt+em|k:iso | 9.20 | 9.27 | 11.05 | 0.587 |
| grec_testB | g:platt|k:multi | 10.25 | 10.28 | 11.65 | - |
| grec_testB | g:platt+em|k:multi | 9.07 | 9.16 | 11.04 | 0.587 |
| grec_testB | g:iso|k:platt | 10.59 | 10.65 | 11.92 | - |
| grec_testB | g:iso+em|k:platt | 9.62 | 9.64 | 11.33 | 0.59 |
| grec_testB | g:iso|k:iso | 10.42 | 10.51 | 11.85 | - |
| grec_testB | g:iso+em|k:iso | 9.46 | 9.48 | 11.27 | 0.59 |
| grec_testB | g:iso|k:multi | 10.32 | 10.41 | 11.85 | - |
| grec_testB | g:iso+em|k:multi | 9.36 | 9.38 | 11.26 | 0.59 |
| grec_testB | kappa_only_raw | 33.83 | 33.83 | 29.06 | - |

Bootstrap (image clusters): 
- refcoco_val: raw ECE 0.92 [0.79, 1.49]; fixed 0.54 [0.45, 1.07]; paired diff -0.38 [-0.68, -0.06]; κ̂ AUROC 90.1 [88.4, 92.0]
- refcocog_test: raw ECE 1.81 [1.52, 2.39]; fixed 0.66 [0.53, 1.34]; paired diff -1.15 [-1.43, -0.57]; κ̂ AUROC 89.0 [87.7, 90.2]

## Presence bootstrap CIs (image clusters, raw ĝ unless noted)

- grec_val: AUROC 98.0 [97.8, 98.2]; N-acc 86.0 [85.3, 86.7]; ECE raw 3.64 [3.37, 3.94] → Platt 2.99 [2.74, 3.31] (paired -0.64 [-0.66, -0.54]); subset ECE raw 7.52 [7.04, 8.00] → Platt 6.52 [6.05, 7.00]
- grec_testA: AUROC 93.7 [93.1, 94.2]; N-acc 79.3 [78.1, 80.5]; ECE raw 4.54 [4.17, 5.13] → Platt 4.70 [4.30, 5.25] (paired 0.16 [-0.01, 0.27]); subset ECE raw 4.54 [4.17, 5.13] → Platt 4.70 [4.30, 5.25]
- grec_testB: AUROC 88.2 [87.0, 89.6]; N-acc 74.0 [72.7, 75.3]; ECE raw 9.22 [8.13, 10.42] → Platt 9.16 [8.07, 10.38] (paired -0.07 [-0.24, 0.08]); subset ECE raw 9.22 [8.13, 10.42] → Platt 9.16 [8.07, 10.38]
- venus: AUROC 77.7 [75.9, 79.3]; N-acc 83.8 [81.2, 86.3]; ECE raw 35.10 [33.67, 36.52] → Platt 21.52 [20.51, 22.51] (paired -13.58 [-14.32, -12.90]); subset ECE raw 35.10 [33.67, 36.52] → Platt 21.52 [20.51, 22.51]

## Risk-controlled accept/abstain (δ=0.1; 50/50 image splits ×reps; coverage = fraction answered)


**refcoco_val** (n=10834, absent 0.0%, correct-if-answered-all 94.9%); default ĝ≥.5: coverage 99.2, selective risk 5.1, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 1.9 | 0.10 | 0.07 |
| selective[g] | 0.02 | 42.9 | 1.27 | 0.085 |
| selective[g] | 0.05 | 90.6 | 4.20 | 0.13 |
| selective[g] | 0.1 | 100.0 | 5.16 | 0.0 |
| joint[g] | 0.01 | 53.8 | 0.85 | 0.255 |
| joint[g] | 0.02 | 71.2 | 1.75 | 0.255 |
| joint[g] | 0.05 | 96.0 | 4.54 | 0.24 |
| joint[g] | 0.1 | 100.0 | 5.16 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 83.1 | 3.43 | False |
| selective_transfer[g] | 0.1 | 99.6 | 5.09 | False |
| selective[g*k] | 0.01 | 48.2 | 0.58 | 0.115 |
| selective[g*k] | 0.02 | 83.3 | 1.51 | 0.1 |
| selective[g*k] | 0.05 | 97.7 | 4.24 | 0.14 |
| selective[g*k] | 0.1 | 100.0 | 5.16 | 0.0 |
| joint[g*k] | 0.01 | 74.6 | 0.83 | 0.26 |
| joint[g*k] | 0.02 | 88.2 | 1.73 | 0.22 |
| joint[g*k] | 0.05 | 98.6 | 4.54 | 0.245 |
| joint[g*k] | 0.1 | 100.0 | 5.16 | 0.0 |
| selective_transfer[g*k] | 0.01 | 63.3 | 0.74 | False |
| selective_transfer[g*k] | 0.02 | 79.8 | 1.28 | False |
| selective_transfer[g*k] | 0.05 | 93.9 | 2.96 | False |
| selective_transfer[g*k] | 0.1 | 99.6 | 5.07 | False |

AURC ĝ 1.90 vs ĝκ̂ 0.88; paired diff -1.02 [-1.30, -0.74]

**refcocog_test** (n=9602, absent 0.0%, correct-if-answered-all 92.5%); default ĝ≥.5: coverage 98.4, selective risk 7.1, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 5.0 | 0.41 | 0.045 |
| selective[g] | 0.05 | 70.3 | 4.13 | 0.075 |
| selective[g] | 0.1 | 100.0 | 7.49 | 0.0 |
| joint[g] | 0.01 | 37.6 | 0.81 | 0.215 |
| joint[g] | 0.02 | 56.6 | 1.75 | 0.205 |
| joint[g] | 0.05 | 83.9 | 4.58 | 0.185 |
| joint[g] | 0.1 | 100.0 | 7.49 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 64.7 | 3.69 | False |
| selective_transfer[g] | 0.1 | 99.1 | 7.24 | False |
| selective[g*k] | 0.01 | 22.9 | 0.38 | 0.04 |
| selective[g*k] | 0.02 | 66.2 | 1.44 | 0.075 |
| selective[g*k] | 0.05 | 90.9 | 4.27 | 0.08 |
| selective[g*k] | 0.1 | 100.0 | 7.48 | 0.0 |
| joint[g*k] | 0.01 | 63.5 | 0.80 | 0.165 |
| joint[g*k] | 0.02 | 77.6 | 1.73 | 0.16 |
| joint[g*k] | 0.05 | 93.1 | 4.56 | 0.185 |
| joint[g*k] | 0.1 | 100.0 | 7.49 | 0.0 |
| selective_transfer[g*k] | 0.01 | 49.3 | 0.76 | False |
| selective_transfer[g*k] | 0.02 | 67.9 | 1.53 | False |
| selective_transfer[g*k] | 0.05 | 90.4 | 4.13 | False |
| selective_transfer[g*k] | 0.1 | 98.9 | 7.07 | False |

AURC ĝ 3.09 vs ĝκ̂ 1.50; paired diff -1.59 [-1.87, -1.34]

**grec_val** (n=25063, absent 35.5%, correct-if-answered-all 57.8%); default ĝ≥.5: coverage 68.3, selective risk 16.6, false-accept on absent 14.0

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.01 | 0.005 |
| selective[g] | 0.02 | 4.2 | 0.59 | 0.08 |
| selective[g] | 0.05 | 32.8 | 4.18 | 0.08 |
| selective[g] | 0.1 | 50.5 | 9.08 | 0.085 |
| joint[g] | 0.01 | 28.2 | 0.88 | 0.205 |
| joint[g] | 0.02 | 36.0 | 1.81 | 0.215 |
| joint[g] | 0.05 | 51.0 | 4.71 | 0.2 |
| joint[g] | 0.1 | 65.4 | 9.48 | 0.135 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 43.9 | 7.34 | True |
| selective_transfer[g] | 0.1 | 70.5 | 18.61 | True |
| selective[g*k] | 0.01 | 0.1 | 0.02 | 0.015 |
| selective[g*k] | 0.02 | 15.3 | 1.19 | 0.04 |
| selective[g*k] | 0.05 | 42.0 | 4.26 | 0.045 |
| selective[g*k] | 0.1 | 55.6 | 9.13 | 0.085 |
| joint[g*k] | 0.01 | 32.3 | 0.88 | 0.155 |
| joint[g*k] | 0.02 | 42.2 | 1.83 | 0.185 |
| joint[g*k] | 0.05 | 54.4 | 4.67 | 0.205 |
| joint[g*k] | 0.1 | 65.3 | 9.49 | 0.22 |
| selective_transfer[g*k] | 0.01 | 29.7 | 2.35 | True |
| selective_transfer[g*k] | 0.02 | 39.9 | 3.78 | True |
| selective_transfer[g*k] | 0.05 | 56.7 | 9.72 | True |
| selective_transfer[g*k] | 0.1 | 70.4 | 18.48 | True |
| absent_conformal[g] | 0.01 | present accepted 56.6 | false-accept absent 0.99 | |
| absent_conformal[g] | 0.02 | present accepted 73.3 | false-accept absent 2.02 | |
| absent_conformal[g] | 0.05 | present accepted 89.6 | false-accept absent 4.99 | |
| absent_conformal[g] | 0.1 | present accepted 96.5 | false-accept absent 10.02 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 87.6 | false-accept absent 4.28 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 95.3 | false-accept absent 8.55 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 98.5 | false-accept absent 15.33 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 99.5 | false-accept absent 23.36 | |

AURC ĝ 13.66 vs ĝκ̂ 12.93; paired diff -0.73 [-0.96, -0.54]

**grec_testA** (n=19200, absent 23.2%, correct-if-answered-all 60.2%); default ĝ≥.5: coverage 78.3, selective risk 24.8, false-accept on absent 20.7

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 11.4 | 3.32 | 0.06 |
| selective[g] | 0.1 | 27.5 | 8.58 | 0.13 |
| joint[g] | 0.01 | 17.0 | 0.87 | 0.26 |
| joint[g] | 0.02 | 24.4 | 1.80 | 0.255 |
| joint[g] | 0.05 | 36.7 | 4.68 | 0.24 |
| joint[g] | 0.1 | 52.1 | 9.55 | 0.24 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 46.5 | 16.44 | True |
| selective_transfer[g] | 0.1 | 80.3 | 25.98 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 27.0 | 3.92 | 0.055 |
| selective[g*k] | 0.1 | 42.1 | 8.88 | 0.11 |
| joint[g*k] | 0.01 | 24.1 | 0.86 | 0.11 |
| joint[g*k] | 0.02 | 33.7 | 1.81 | 0.17 |
| joint[g*k] | 0.05 | 45.4 | 4.65 | 0.24 |
| joint[g*k] | 0.1 | 58.3 | 9.45 | 0.135 |
| selective_transfer[g*k] | 0.01 | 27.0 | 3.85 | True |
| selective_transfer[g*k] | 0.02 | 37.9 | 6.89 | True |
| selective_transfer[g*k] | 0.05 | 63.9 | 18.90 | True |
| selective_transfer[g*k] | 0.1 | 80.0 | 25.75 | True |
| absent_conformal[g] | 0.01 | present accepted 16.8 | false-accept absent 1.01 | |
| absent_conformal[g] | 0.02 | present accepted 30.1 | false-accept absent 1.95 | |
| absent_conformal[g] | 0.05 | present accepted 61.5 | false-accept absent 5.00 | |
| absent_conformal[g] | 0.1 | present accepted 84.0 | false-accept absent 9.99 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 83.4 | false-accept absent 9.85 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 92.2 | false-accept absent 15.67 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 96.1 | false-accept absent 22.01 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 97.8 | false-accept absent 29.16 | |

AURC ĝ 17.26 vs ĝκ̂ 14.87; paired diff -2.39 [-2.75, -2.02]

**grec_testB** (n=16063, absent 29.1%, correct-if-answered-all 52.5%); default ĝ≥.5: coverage 72.6, selective risk 30.5, false-accept on absent 26.0

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.1 | 0.07 | 0.01 |
| selective[g] | 0.1 | 14.5 | 7.66 | 0.075 |
| joint[g] | 0.01 | 12.3 | 0.87 | 0.23 |
| joint[g] | 0.02 | 18.5 | 1.79 | 0.24 |
| joint[g] | 0.05 | 29.3 | 4.70 | 0.29 |
| joint[g] | 0.1 | 43.2 | 9.49 | 0.25 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 39.5 | 20.57 | True |
| selective_transfer[g] | 0.1 | 75.0 | 31.92 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.04 | 0.005 |
| selective[g*k] | 0.1 | 20.1 | 8.22 | 0.065 |
| joint[g*k] | 0.01 | 14.4 | 0.87 | 0.195 |
| joint[g*k] | 0.02 | 20.8 | 1.79 | 0.18 |
| joint[g*k] | 0.05 | 33.8 | 4.61 | 0.095 |
| joint[g*k] | 0.1 | 48.3 | 9.46 | 0.225 |
| selective_transfer[g*k] | 0.01 | 18.9 | 7.79 | True |
| selective_transfer[g*k] | 0.02 | 29.6 | 11.48 | True |
| selective_transfer[g*k] | 0.05 | 55.5 | 22.78 | True |
| selective_transfer[g*k] | 0.1 | 74.4 | 31.54 | True |
| absent_conformal[g] | 0.01 | present accepted 11.0 | false-accept absent 1.00 | |
| absent_conformal[g] | 0.02 | present accepted 20.4 | false-accept absent 1.95 | |
| absent_conformal[g] | 0.05 | present accepted 41.4 | false-accept absent 4.99 | |
| absent_conformal[g] | 0.1 | present accepted 67.4 | false-accept absent 9.97 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 78.5 | false-accept absent 13.65 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 87.9 | false-accept absent 19.97 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 92.1 | false-accept absent 27.50 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 94.0 | false-accept absent 34.32 | |

AURC ĝ 23.77 vs ĝκ̂ 21.89; paired diff -1.88 [-2.43, -1.48]

**ss2** (n=1272, absent 0.0%, correct-if-answered-all 90.1%); default ĝ≥.5: coverage 89.6, selective risk 8.5, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.4 | 0.07 | 0.01 |
| selective[g] | 0.1 | 70.1 | 6.69 | 0.07 |
| joint[g] | 0.01 | 14.7 | 0.48 | 0.125 |
| joint[g] | 0.02 | 30.2 | 1.23 | 0.145 |
| joint[g] | 0.05 | 63.8 | 3.98 | 0.2 |
| joint[g] | 0.1 | 94.2 | 8.58 | 0.185 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 3.3 | 4.76 | False |
| selective_transfer[g] | 0.1 | 15.2 | 2.59 | False |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 21.0 | 1.41 | 0.025 |
| selective[g*k] | 0.1 | 92.7 | 7.22 | 0.045 |
| joint[g*k] | 0.01 | 22.4 | 0.47 | 0.05 |
| joint[g*k] | 0.02 | 46.5 | 1.16 | 0.095 |
| joint[g*k] | 0.05 | 78.0 | 4.00 | 0.205 |
| joint[g*k] | 0.1 | 97.5 | 8.57 | 0.18 |
| selective_transfer[g*k] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.02 | 10.1 | 0.78 | False |
| selective_transfer[g*k] | 0.05 | 68.1 | 3.93 | False |
| selective_transfer[g*k] | 0.1 | 92.3 | 6.73 | False |

AURC ĝ 5.33 vs ĝκ̂ 3.29; paired diff -2.04 [-2.93, -1.29]

**venus** (n=6165, absent 14.6%, correct-if-answered-all 43.5%); default ĝ≥.5: coverage 51.7, selective risk 44.9, false-accept on absent 16.2

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 4.0 | 0.80 | 0.195 |
| joint[g] | 0.02 | 6.4 | 1.64 | 0.175 |
| joint[g] | 0.05 | 14.3 | 4.46 | 0.215 |
| joint[g] | 0.1 | 25.4 | 9.15 | 0.145 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.1 | 40.00 | True |
| selective_transfer[g] | 0.1 | 2.1 | 17.32 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g*k] | 0.01 | 4.7 | 0.80 | 0.215 |
| joint[g*k] | 0.02 | 8.6 | 1.64 | 0.165 |
| joint[g*k] | 0.05 | 19.3 | 4.46 | 0.115 |
| joint[g*k] | 0.1 | 33.2 | 9.05 | 0.12 |
| selective_transfer[g*k] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.02 | 1.5 | 7.61 | True |
| selective_transfer[g*k] | 0.05 | 24.3 | 24.97 | True |
| selective_transfer[g*k] | 0.1 | 50.4 | 34.93 | True |
| absent_conformal[g] | 0.01 | present accepted 7.0 | false-accept absent 0.89 | |
| absent_conformal[g] | 0.02 | present accepted 11.1 | false-accept absent 1.84 | |
| absent_conformal[g] | 0.05 | present accepted 23.7 | false-accept absent 4.69 | |
| absent_conformal[g] | 0.1 | present accepted 38.8 | false-accept absent 9.72 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 1.3 | false-accept absent 0.00 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 2.2 | false-accept absent 0.11 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 4.3 | false-accept absent 0.22 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 9.9 | false-accept absent 1.45 | |

AURC ĝ 42.46 vs ĝκ̂ 35.02; paired diff -7.44 [-8.39, -6.57]

## Prop. 1 two-knob plug-in rule vs one-knob thresholds (expected cost per query, lower is better)

| set | costs | two-knob plug-in | ĝ≥.5 | ĝ thr tuned on calib | ĝκ̂ thr tuned | ĝ oracle thr | paired two−one(tuned) |
|---|---|---|---|---|---|---|---|
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0563 | 0.0542 | 0.0588 | 0.0545 | 0.0518 | -0.0026 [-0.0040, -0.0011] |
| refcoco_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.0846 | 0.1043 | 0.1100 | 0.0853 | 0.1031 | -0.0253 [-0.0307, -0.0199] |
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.0599 | 0.0582 | 0.0592 | 0.0575 | 0.0522 | 0.0007 [0.0003, 0.0013] |
| refcoco_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0601 | 0.0542 | 0.0613 | 0.0567 | 0.0518 | -0.0012 [-0.0025, 0.0003] |
| refcoco_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1230 | 0.2045 | 0.2032 | 0.1288 | 0.2027 | -0.0803 [-0.0942, -0.0653] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0818 | 0.0778 | 0.0856 | 0.0790 | 0.0753 | -0.0038 [-0.0057, -0.0021] |
| refcocog_test | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1245 | 0.1476 | 0.1547 | 0.1250 | 0.1473 | -0.0302 [-0.0367, -0.0242] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.0906 | 0.0859 | 0.0882 | 0.0855 | 0.0759 | 0.0024 [0.0014, 0.0035] |
| refcocog_test | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0886 | 0.0778 | 0.0902 | 0.0829 | 0.0753 | -0.0017 [-0.0036, 0.0002] |
| refcocog_test | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1723 | 0.2872 | 0.2681 | 0.1816 | 0.2681 | -0.0959 [-0.1116, -0.0810] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1046 | 0.1191 | 0.1053 | 0.1072 | 0.1031 | -0.0007 [-0.0025, 0.0012] |
| grec_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1312 | 0.1825 | 0.1604 | 0.1315 | 0.1561 | -0.0292 [-0.0342, -0.0240] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1213 | 0.1249 | 0.1227 | 0.1282 | 0.1190 | -0.0014 [-0.0024, -0.0002] |
| grec_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1261 | 0.1688 | 0.1301 | 0.1361 | 0.1211 | -0.0040 [-0.0064, -0.0016] |
| grec_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1569 | 0.3095 | 0.2587 | 0.1571 | 0.2587 | -0.1018 [-0.1134, -0.0908] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2020 | 0.2105 | 0.2036 | 0.2024 | 0.2024 | -0.0017 [-0.0045, 0.0009] |
| grec_testA | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2666 | 0.3564 | 0.3353 | 0.2726 | 0.3299 | -0.0687 [-0.0785, -0.0600] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2271 | 0.2272 | 0.2268 | 0.2327 | 0.2263 | 0.0003 [-0.0009, 0.0015] |
| grec_testA | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2322 | 0.2586 | 0.2360 | 0.2368 | 0.2298 | -0.0038 [-0.0072, -0.0007] |
| grec_testA | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2942 | 0.6482 | 0.5771 | 0.3297 | 0.5771 | -0.2829 [-0.3081, -0.2581] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2322 | 0.2506 | 0.2378 | 0.2343 | 0.2334 | -0.0057 [-0.0088, -0.0025] |
| grec_testB | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2806 | 0.3963 | 0.3664 | 0.2845 | 0.3535 | -0.0858 [-0.0982, -0.0739] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2804 | 0.2799 | 0.2790 | 0.2866 | 0.2778 | 0.0014 [-0.0002, 0.0030] |
| grec_testB | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2796 | 0.3264 | 0.2898 | 0.2864 | 0.2745 | -0.0102 [-0.0139, -0.0063] |
| grec_testB | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.3119 | 0.6875 | 0.5936 | 0.3332 | 0.5936 | -0.2818 [-0.3107, -0.2506] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1549 | 0.1281 | 0.1372 | 0.1171 | 0.0987 | 0.0177 [0.0072, 0.0269] |
| ss2 | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2362 | 0.2044 | 0.2170 | 0.1836 | 0.1938 | 0.0193 [-0.0020, 0.0388] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1376 | 0.1800 | 0.1431 | 0.1226 | 0.0991 | -0.0055 [-0.0137, 0.0024] |
| ss2 | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2131 | 0.1281 | 0.1619 | 0.1333 | 0.0987 | 0.0511 [0.0402, 0.0620] |
| ss2 | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.3113 | 0.3569 | 0.3365 | 0.2838 | 0.3263 | -0.0252 [-0.0621, 0.0070] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.3677 | 0.4128 | 0.4086 | 0.3650 | 0.3960 | -0.0409 [-0.0481, -0.0341] |
| venus | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.4127 | 0.6216 | 0.5184 | 0.4363 | 0.4268 | -0.1057 [-0.1216, -0.0900] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.5713 | 0.5932 | 0.5598 | 0.5617 | 0.5314 | 0.0115 [0.0058, 0.0170] |
| venus | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.3929 | 0.4365 | 0.4123 | 0.3858 | 0.4040 | -0.0195 [-0.0268, -0.0129] |
| venus | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.4453 | 1.0391 | 0.5654 | 0.4797 | 0.4537 | -0.1200 [-0.1459, -0.0963] |
