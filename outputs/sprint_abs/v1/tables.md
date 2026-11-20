# sprint_abs tables (v1)

## GREC official protocol (gRefCOCO; Pr@(F1=1, GIoU>=0.5), N-acc, T-acc)

| split | variant | Pr@F1=1 | 95% CI | N-acc | T-acc | Pr single | Pr multi | Pr (plain IoU) |
|---|---|---|---|---|---|---|---|---|
| grec_val | single_raw@0.7(official) | 71.6 | 71.6 [71.1, 72.1] | 91.4 | 95.1 | 90.4 | 0.0 | 71.8 |
| grec_val | single_raw@0.5 | 70.6 | 70.6 [70.0, 71.1] | 87.4 | 97.4 | 91.4 | 0.0 | 70.8 |
| grec_val | single_platt@0.7 | 71.4 | 71.4 [70.9, 71.9] | 90.6 | 95.8 | 90.8 | 0.0 | 71.7 |
| grec_val | multi_platt@0.7(official) | 71.4 | 71.4 [70.9, 71.9] | 90.6 | 95.8 | 90.8 | 0.0 | 71.7 |
| grec_val | multi_raw@0.7(official) | 71.6 | 71.6 [71.1, 72.1] | 91.4 | 95.1 | 90.4 | 0.0 | 71.8 |
| grec_val | multi_platt@tuned | 69.3 | 69.3 [68.7, 69.8] | 82.2 | 98.5 | 91.5 | 2.3 | 69.6 |
| grec_val | single_platt@tuned | 71.1 | 71.1 [70.6, 71.6] | 89.5 | 96.3 | 91.0 | 0.0 | 71.4 |
| grec_val | paired diff multi-single (official) | -0.1 [-0.2, -0.1] | | | | | | |
| grec_testA | single_raw@0.7(official) | 47.5 | 47.5 [46.8, 48.3] | 85.4 | 91.4 | 90.1 | 0.0 | 47.7 |
| grec_testA | single_raw@0.5 | 46.7 | 46.7 [46.0, 47.4] | 80.8 | 94.4 | 90.7 | 0.0 | 46.8 |
| grec_testA | single_platt@0.7 | 47.4 | 47.4 [46.6, 48.1] | 84.4 | 92.4 | 90.2 | 0.0 | 47.5 |
| grec_testA | multi_platt@0.7(official) | 47.4 | 47.4 [46.6, 48.1] | 84.4 | 92.4 | 90.2 | 0.0 | 47.5 |
| grec_testA | multi_raw@0.7(official) | 47.5 | 47.5 [46.8, 48.3] | 85.4 | 91.4 | 90.1 | 0.0 | 47.7 |
| grec_testA | multi_platt@tuned | 46.2 | 46.2 [45.5, 46.9] | 76.1 | 96.4 | 90.7 | 1.3 | 46.4 |
| grec_testA | single_platt@tuned | 47.1 | 47.1 [46.4, 47.9] | 83.1 | 93.3 | 90.5 | 0.0 | 47.2 |
| grec_testA | paired diff multi-single (official) | -0.2 [-0.3, -0.1] | | | | | | |
| grec_testB | single_raw@0.7(official) | 51.3 | 51.3 [50.3, 52.4] | 81.9 | 87.4 | 78.3 | 0.0 | 51.7 |
| grec_testB | single_raw@0.5 | 50.3 | 50.3 [49.2, 51.3] | 77.1 | 90.1 | 79.2 | 0.0 | 50.6 |
| grec_testB | single_platt@0.7 | 51.0 | 51.0 [50.0, 52.1] | 80.5 | 88.3 | 78.6 | 0.0 | 51.4 |
| grec_testB | multi_platt@0.7(official) | 51.0 | 51.0 [50.0, 52.1] | 80.5 | 88.3 | 78.6 | 0.0 | 51.4 |
| grec_testB | multi_raw@0.7(official) | 51.3 | 51.3 [50.3, 52.4] | 81.9 | 87.4 | 78.3 | 0.0 | 51.7 |
| grec_testB | multi_platt@tuned | 49.1 | 49.1 [48.1, 50.2] | 71.3 | 91.8 | 79.3 | 1.3 | 49.6 |
| grec_testB | single_platt@tuned | 50.8 | 50.8 [49.8, 51.9] | 79.5 | 89.1 | 78.9 | 0.0 | 51.2 |
| grec_testB | paired diff multi-single (official) | -0.3 [-0.4, -0.2] | | | | | | |
| grefcalib | single_raw@0.7(official) | 62.4 | - | 97.6 | 96.7 | 89.9 | 0.0 | 63.0 |
| grefcalib | single_raw@0.5 | 62.6 | - | 95.8 | 98.2 | 90.4 | 0.0 | 63.1 |
| grefcalib | single_platt@0.7 | 62.6 | - | 97.3 | 97.3 | 90.3 | 0.0 | 63.2 |
| grefcalib | multi_platt@0.7(official) | 62.6 | - | 97.3 | 97.3 | 90.3 | 0.0 | 63.2 |
| grefcalib | multi_raw@0.7(official) | 62.4 | - | 97.6 | 96.7 | 89.9 | 0.0 | 63.0 |
| grefcalib | multi_platt@tuned | 63.2 | - | 93.8 | 99.1 | 90.5 | 2.6 | 64.0 |
| grefcalib | single_platt@tuned | 62.6 | - | 97.1 | 97.7 | 90.3 | 0.0 | 63.2 |

Tuned on grefcalib: {"pr_f1_grefcalib": 0.6319523918516823, "t_presence": 0.35, "t_extra": 0.25, "single_t_presence": 0.6499999999999999, "single_pr_f1_grefcalib": 0.6264591439688716}; extra-box model: {"n": 2226, "pos": 0.1347708894878706}

## VenusBench-GD, official UI-Venus evaluator

| config | Element | Visual | Spatial | Basic | Reasoning | Functional | Refusal | Advanced | Overall | n |
|---|---|---|---|---|---|---|---|---|---|---|
| platt_gated | 52.86 | 65.44 | 46.45 | 54.00 | 11.02 | 33.00 | 52.84 | 30.60 | 43.73 | 6165 |
| raw_gated | 42.36 | 51.37 | 35.08 | 42.38 | 7.41 | 26.71 | 81.76 | 37.10 | 40.06 | 6165 |
| raw_ungated | 57.89 | 71.51 | 55.30 | 60.42 | 13.28 | 36.14 | 81.76 | 41.94 | 52.31 | 6165 |
| stack_gated | 53.61 | 69.61 | 53.26 | 57.39 | 6.14 | 31.43 | 33.93 | 21.91 | 41.82 | 6165 |

## Presence ĝ calibration (fit on held-out training-distribution splits)

| set | subset | method | AUROC | N-acc@.5 | recall@.5 | ECE15 | ECE15-mass | ECE10 | Brier | NLL | +EM prior: ECE15 / N-acc / π̂ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grec_val | all | raw | 97.9 | 87.4 | 97.4 | 2.3 | 2.2 | 2.3 | 4.75 | 0.174 | - |
| grec_val | all | temp | 97.9 | 87.4 | 97.4 | 2.4 | 2.4 | 2.4 | 4.77 | 0.177 | 1.6 / 89.1 / 0.657 |
| grec_val | all | platt | 97.9 | 86.1 | 97.8 | 3.0 | 3.0 | 3.0 | 4.90 | 0.182 | 2.2 / 87.6 / 0.666 |
| grec_val | all | beta | 97.9 | 86.1 | 97.7 | 3.0 | 3.0 | 3.0 | 4.90 | 0.182 | 2.2 / 87.6 / 0.666 |
| grec_val | all | iso | 97.9 | 86.9 | 97.5 | 2.9 | 2.9 | 2.9 | 4.90 | 0.181 | 2.1 / 88.3 / 0.665 |
| grec_val | all | multi | 97.9 | 85.5 | 97.8 | 3.0 | 3.0 | 3.0 | 4.92 | 0.174 | 1.8 / 87.3 / 0.663 |
| grec_val | gref_only | raw | 96.4 | 87.4 | 94.9 | 5.9 | 5.9 | 5.9 | 7.53 | 0.275 | - |
| grec_val | gref_only | temp | 96.4 | 87.4 | 94.9 | 5.9 | 5.9 | 5.9 | 7.56 | 0.280 | 3.7 / 93.6 / 0.359 |
| grec_val | gref_only | platt | 96.4 | 86.1 | 95.5 | 7.0 | 7.0 | 7.0 | 7.91 | 0.292 | 2.9 / 92.8 / 0.371 |
| grec_val | gref_only | beta | 96.4 | 86.1 | 95.5 | 7.0 | 7.0 | 7.0 | 7.91 | 0.292 | 2.9 / 92.9 / 0.371 |
| grec_val | gref_only | iso | 96.4 | 86.9 | 95.1 | 6.8 | 6.8 | 6.8 | 7.88 | 0.291 | 2.5 / 91.4 / 0.374 |
| grec_val | gref_only | multi | 96.4 | 85.5 | 95.7 | 7.5 | 7.5 | 7.5 | 7.94 | 0.277 | 2.9 / 93.8 / 0.354 |
| grec_testA | all | raw | 94.1 | 80.8 | 94.4 | 4.4 | 4.3 | 4.4 | 6.95 | 0.299 | - |
| grec_testA | all | temp | 94.1 | 80.8 | 94.4 | 4.6 | 4.4 | 4.6 | 6.98 | 0.306 | 4.3 / 79.6 / 0.766 |
| grec_testA | all | platt | 94.1 | 79.5 | 94.9 | 4.2 | 4.1 | 4.2 | 6.86 | 0.300 | 4.2 / 78.0 / 0.775 |
| grec_testA | all | beta | 94.1 | 79.5 | 94.9 | 4.2 | 4.1 | 4.2 | 6.86 | 0.300 | 4.2 / 78.0 / 0.775 |
| grec_testA | all | iso | 94.1 | 80.1 | 94.6 | 4.6 | 4.4 | 4.6 | 6.98 | 0.292 | 4.6 / 79.5 / 0.774 |
| grec_testA | all | multi | 94.0 | 78.6 | 95.3 | 3.2 | 3.2 | 3.1 | 6.78 | 0.271 | 3.3 / 76.7 / 0.776 |
| grec_testA | gref_only | raw | 93.2 | 80.8 | 93.1 | 4.9 | 5.0 | 4.9 | 8.54 | 0.327 | - |
| grec_testA | gref_only | temp | 93.2 | 80.8 | 93.1 | 5.1 | 5.2 | 5.1 | 8.57 | 0.334 | 5.8 / 82.6 / 0.656 |
| grec_testA | gref_only | platt | 93.2 | 79.5 | 94.0 | 4.9 | 4.8 | 4.9 | 8.48 | 0.332 | 5.0 / 80.8 / 0.669 |
| grec_testA | gref_only | beta | 93.2 | 79.5 | 94.0 | 4.9 | 4.8 | 4.9 | 8.48 | 0.333 | 5.0 / 80.9 / 0.669 |
| grec_testA | gref_only | iso | 93.2 | 80.1 | 93.4 | 5.5 | 5.4 | 5.5 | 8.62 | 0.329 | 5.5 / 81.4 / 0.669 |
| grec_testA | gref_only | multi | 93.1 | 78.6 | 94.5 | 3.6 | 3.6 | 3.5 | 8.38 | 0.307 | 3.9 / 80.2 / 0.666 |
| grec_testB | all | raw | 88.7 | 77.1 | 90.1 | 8.4 | 8.2 | 8.4 | 11.63 | 0.614 | - |
| grec_testB | all | temp | 88.7 | 77.1 | 90.1 | 8.7 | 8.4 | 8.7 | 11.67 | 0.634 | 8.5 / 78.1 / 0.688 |
| grec_testB | all | platt | 88.7 | 75.0 | 90.6 | 8.9 | 8.8 | 8.9 | 11.66 | 0.620 | 8.7 / 76.1 / 0.699 |
| grec_testB | all | beta | 88.7 | 75.0 | 90.6 | 8.9 | 8.8 | 8.9 | 11.66 | 0.619 | 8.7 / 76.2 / 0.699 |
| grec_testB | all | iso | 88.7 | 76.4 | 90.3 | 9.2 | 8.9 | 9.2 | 11.75 | 0.587 | 9.2 / 77.0 / 0.698 |
| grec_testB | all | multi | 88.6 | 74.2 | 90.7 | 7.7 | 7.7 | 7.7 | 11.55 | 0.536 | 7.5 / 75.3 / 0.694 |
| grec_testB | gref_only | raw | 89.8 | 77.1 | 91.5 | 8.2 | 7.6 | 8.2 | 12.09 | 0.479 | - |
| grec_testB | gref_only | temp | 89.8 | 77.1 | 91.5 | 8.5 | 7.8 | 8.5 | 12.15 | 0.492 | 8.0 / 81.0 / 0.557 |
| grec_testB | gref_only | platt | 89.8 | 75.0 | 92.2 | 8.9 | 8.5 | 8.9 | 12.23 | 0.496 | 7.6 / 79.7 / 0.572 |
| grec_testB | gref_only | beta | 89.8 | 75.0 | 92.2 | 8.8 | 8.5 | 8.8 | 12.23 | 0.497 | 7.6 / 79.8 / 0.572 |
| grec_testB | gref_only | iso | 89.8 | 76.4 | 91.7 | 9.2 | 8.5 | 9.2 | 12.34 | 0.490 | 8.3 / 79.8 / 0.574 |
| grec_testB | gref_only | multi | 89.8 | 74.2 | 92.5 | 7.6 | 7.6 | 7.6 | 12.09 | 0.448 | 6.5 / 80.1 / 0.559 |
| calib_natural | all | raw | 99.4 | 95.0 | 98.3 | 0.7 | 0.6 | 0.6 | 2.03 | 0.076 | - |
| calib_natural | all | temp | 99.4 | 95.0 | 98.3 | 0.6 | 0.6 | 0.5 | 2.03 | 0.076 | 0.6 / 95.0 / 0.718 |
| calib_natural | all | platt | 99.4 | 94.2 | 98.7 | 0.5 | 0.3 | 0.5 | 2.02 | 0.076 | 0.5 / 94.2 / 0.723 |
| calib_natural | all | beta | 99.4 | 94.2 | 98.7 | 0.5 | 0.3 | 0.5 | 2.02 | 0.076 | 0.5 / 94.2 / 0.723 |
| calib_natural | all | iso | 99.4 | 94.9 | 98.4 | 0.3 | 0.2 | 0.2 | 2.00 | 0.075 | 0.3 / 94.9 / 0.723 |
| calib_natural | all | multi | 99.4 | 94.1 | 98.7 | 1.2 | 1.0 | 1.2 | 2.04 | 0.078 | 1.2 / 94.2 / 0.721 |
| grefcalib | all | raw | 99.4 | 95.8 | 98.2 | 2.7 | 2.8 | 2.7 | 1.61 | 0.063 | - |
| grefcalib | all | temp | 99.4 | 95.8 | 98.2 | 2.6 | 2.7 | 2.6 | 1.60 | 0.062 | 0.8 / 92.7 / 0.890 |
| grefcalib | all | platt | 99.4 | 95.1 | 98.5 | 2.2 | 2.2 | 2.2 | 1.45 | 0.057 | 0.6 / 92.0 / 0.893 |
| grefcalib | all | beta | 99.4 | 95.1 | 98.5 | 2.2 | 2.2 | 2.2 | 1.45 | 0.057 | 0.6 / 92.0 / 0.893 |
| grefcalib | all | iso | 99.4 | 95.8 | 98.3 | 2.2 | 2.2 | 2.2 | 1.47 | 0.057 | 0.5 / 92.2 / 0.893 |
| grefcalib | all | multi | 99.4 | 95.1 | 98.6 | 2.7 | 2.8 | 2.7 | 1.46 | 0.060 | 0.9 / 90.7 / 0.894 |
| guisyn_test | natural_fit | raw | 85.2 | 75.0 | 83.5 | 13.1 | 13.0 | 13.1 | 14.28 | 0.569 | - |
| guisyn_test | natural_fit | temp | 85.2 | 75.0 | 83.5 | 13.2 | 13.2 | 13.2 | 14.36 | 0.583 | 14.8 / 77.0 / 0.659 |
| guisyn_test | natural_fit | platt | 85.2 | 73.2 | 85.3 | 11.9 | 11.9 | 11.9 | 13.71 | 0.561 | 13.0 / 74.7 / 0.681 |
| guisyn_test | natural_fit | beta | 85.2 | 73.2 | 85.3 | 11.9 | 11.9 | 11.9 | 13.71 | 0.561 | 13.0 / 74.7 / 0.681 |
| guisyn_test | natural_fit | iso | 85.0 | 74.3 | 84.2 | 12.3 | 12.2 | 12.3 | 14.14 | 0.557 | 13.3 / 75.4 / 0.678 |
| guisyn_test | natural_fit | multi | 82.4 | 68.9 | 83.1 | 12.6 | 12.6 | 12.5 | 14.99 | 0.552 | 15.0 / 72.9 / 0.653 |
| guisyn_test | gui_fit | raw | 85.2 | 75.0 | 83.5 | 13.1 | 13.0 | 13.1 | 14.28 | 0.569 | - |
| guisyn_test | gui_fit | temp | 85.2 | 75.0 | 83.5 | 11.7 | 12.0 | 11.8 | 13.56 | 0.429 | 34.0 / 89.0 / 0.407 |
| guisyn_test | gui_fit | platt | 85.2 | 52.5 | 95.8 | 5.7 | 5.7 | 5.6 | 11.71 | 0.389 | 6.3 / 57.9 / 0.726 |
| guisyn_test | gui_fit | beta | 85.2 | 51.6 | 96.0 | 2.5 | 3.4 | 2.7 | 11.41 | 0.375 | 2.5 / 54.3 / 0.732 |
| guisyn_test | gui_fit | iso | 85.2 | 52.5 | 95.8 | 1.3 | 1.8 | 1.3 | 11.31 | 0.369 | 2.0 / 53.3 / 0.734 |
| guisyn_test | gui_fit | multi | 96.4 | 84.3 | 94.6 | 1.3 | 1.2 | 1.3 | 6.21 | 0.209 | 1.2 / 84.6 / 0.757 |
| venus | natural_fit | raw | 73.8 | 81.8 | 53.5 | 37.9 | 37.9 | 37.6 | 33.51 | 1.307 | - |
| venus | natural_fit | temp | 73.8 | 81.8 | 53.5 | 38.0 | 38.0 | 37.6 | 33.77 | 1.348 | 54.5 / 94.3 / 0.310 |
| venus | natural_fit | platt | 73.8 | 79.5 | 56.1 | 35.6 | 35.7 | 35.2 | 31.85 | 1.240 | 51.3 / 92.5 / 0.342 |
| venus | natural_fit | beta | 73.8 | 79.5 | 56.1 | 35.6 | 35.7 | 35.2 | 31.86 | 1.238 | 51.3 / 92.7 / 0.342 |
| venus | natural_fit | iso | 73.8 | 81.0 | 54.5 | 35.8 | 36.1 | 35.6 | 32.85 | 1.202 | 50.2 / 89.9 / 0.354 |
| venus | natural_fit | multi | 75.0 | 81.2 | 56.1 | 35.6 | 35.7 | 35.5 | 30.13 | 1.031 | 57.7 / 95.1 / 0.277 |
| venus | gui_fit | raw | 73.8 | 81.8 | 53.5 | 37.9 | 37.9 | 37.6 | 33.51 | 1.307 | - |
| venus | gui_fit | temp | 73.8 | 81.8 | 53.5 | 36.9 | 36.9 | 36.9 | 27.31 | 0.779 | 85.3 / 100.0 / 0.001 |
| venus | gui_fit | platt | 73.8 | 52.8 | 78.6 | 21.5 | 21.5 | 21.5 | 17.38 | 0.522 | 69.5 / 99.7 / 0.159 |
| venus | gui_fit | beta | 73.8 | 52.1 | 79.5 | 20.2 | 20.2 | 20.2 | 18.02 | 0.564 | 59.2 / 98.3 / 0.263 |
| venus | gui_fit | iso | 73.8 | 52.8 | 78.6 | 20.5 | 20.5 | 20.5 | 18.10 | 0.559 | 54.8 / 92.7 / 0.306 |
| venus | gui_fit | multi | 53.8 | 33.9 | 68.1 | 27.7 | 27.7 | 27.7 | 27.60 | 0.871 | 31.8 / 42.7 / 0.597 |

## κ̂ calibration (natural: event IoU>=0.5; GUI: click in box)

| set | fit | method | AUROC | AURC | ECE15 | ECE15-mass | Brier | mean κ̂ | rate |
|---|---|---|---|---|---|---|---|---|---|
| refcoco_val | natural_fit | raw | 90.1 | 1.11 | 0.6 | 0.5 | 4.54 | 92.8 | 93.2 |
| refcoco_val | natural_fit | temp | 90.1 | 1.11 | 0.6 | 0.4 | 4.54 | 93.1 | 93.2 |
| refcoco_val | natural_fit | platt | 90.1 | 1.11 | 0.5 | 0.4 | 4.54 | 93.2 | 93.2 |
| refcoco_val | natural_fit | beta | 90.1 | 1.11 | 0.5 | 0.4 | 4.55 | 93.2 | 93.2 |
| refcoco_val | natural_fit | iso | 90.1 | 1.11 | 0.7 | 0.8 | 4.55 | 93.2 | 93.2 |
| refcoco_val | natural_fit | multi | 90.0 | 1.14 | 1.0 | 0.6 | 4.52 | 92.6 | 93.2 |
| refcoco_testA | natural_fit | raw | 91.6 | 0.73 | 0.7 | 0.7 | 3.58 | 94.6 | 94.5 |
| refcoco_testA | natural_fit | temp | 91.6 | 0.73 | 0.7 | 0.7 | 3.58 | 94.8 | 94.5 |
| refcoco_testA | natural_fit | platt | 91.6 | 0.73 | 0.7 | 0.7 | 3.58 | 94.9 | 94.5 |
| refcoco_testA | natural_fit | beta | 91.6 | 0.73 | 0.6 | 0.7 | 3.58 | 94.9 | 94.5 |
| refcoco_testA | natural_fit | iso | 91.6 | 0.73 | 0.9 | 0.9 | 3.61 | 94.9 | 94.5 |
| refcoco_testA | natural_fit | multi | 92.0 | 0.71 | 1.0 | 0.9 | 3.55 | 94.3 | 94.5 |
| refcoco_testB | natural_fit | raw | 89.6 | 1.69 | 1.3 | 1.5 | 6.26 | 89.8 | 90.5 |
| refcoco_testB | natural_fit | temp | 89.6 | 1.69 | 1.5 | 1.5 | 6.26 | 90.1 | 90.5 |
| refcoco_testB | natural_fit | platt | 89.6 | 1.69 | 1.6 | 1.5 | 6.26 | 90.2 | 90.5 |
| refcoco_testB | natural_fit | beta | 89.6 | 1.69 | 1.2 | 1.1 | 6.24 | 90.2 | 90.5 |
| refcoco_testB | natural_fit | iso | 89.6 | 1.69 | 1.3 | 1.6 | 6.27 | 90.0 | 90.5 |
| refcoco_testB | natural_fit | multi | 89.6 | 1.69 | 1.4 | 1.3 | 6.23 | 89.6 | 90.5 |
| refcocoplus_val | natural_fit | raw | 90.7 | 1.73 | 1.3 | 1.1 | 6.26 | 88.7 | 89.7 |
| refcocoplus_val | natural_fit | temp | 90.7 | 1.73 | 1.0 | 1.1 | 6.26 | 88.9 | 89.7 |
| refcocoplus_val | natural_fit | platt | 90.7 | 1.73 | 1.1 | 1.0 | 6.25 | 89.1 | 89.7 |
| refcocoplus_val | natural_fit | beta | 90.7 | 1.73 | 1.0 | 0.8 | 6.25 | 89.1 | 89.7 |
| refcocoplus_val | natural_fit | iso | 90.7 | 1.73 | 1.0 | 1.0 | 6.25 | 89.0 | 89.7 |
| refcocoplus_val | natural_fit | multi | 90.7 | 1.74 | 1.2 | 1.1 | 6.21 | 88.7 | 89.7 |
| refcocoplus_testA | natural_fit | raw | 90.8 | 1.01 | 1.1 | 1.1 | 4.44 | 92.6 | 93.3 |
| refcocoplus_testA | natural_fit | temp | 90.8 | 1.01 | 0.8 | 1.0 | 4.44 | 92.8 | 93.3 |
| refcocoplus_testA | natural_fit | platt | 90.8 | 1.01 | 0.8 | 1.0 | 4.43 | 92.9 | 93.3 |
| refcocoplus_testA | natural_fit | beta | 90.8 | 1.01 | 1.0 | 1.0 | 4.45 | 93.0 | 93.3 |
| refcocoplus_testA | natural_fit | iso | 90.8 | 1.11 | 0.9 | 0.9 | 4.42 | 93.0 | 93.3 |
| refcocoplus_testA | natural_fit | multi | 90.8 | 1.02 | 1.1 | 1.2 | 4.43 | 92.4 | 93.3 |
| refcocoplus_testB | natural_fit | raw | 89.0 | 3.29 | 1.7 | 1.8 | 8.88 | 83.0 | 84.5 |
| refcocoplus_testB | natural_fit | temp | 89.0 | 3.29 | 1.5 | 1.6 | 8.89 | 83.3 | 84.5 |
| refcocoplus_testB | natural_fit | platt | 89.0 | 3.29 | 1.2 | 1.4 | 8.87 | 83.6 | 84.5 |
| refcocoplus_testB | natural_fit | beta | 89.0 | 3.29 | 1.2 | 1.5 | 8.86 | 83.6 | 84.5 |
| refcocoplus_testB | natural_fit | iso | 89.0 | 3.29 | 1.5 | 1.6 | 8.89 | 83.3 | 84.5 |
| refcocoplus_testB | natural_fit | multi | 89.0 | 3.29 | 1.5 | 1.5 | 8.86 | 83.3 | 84.5 |
| refcocog_val | natural_fit | raw | 87.4 | 2.11 | 1.2 | 1.4 | 6.76 | 89.3 | 90.0 |
| refcocog_val | natural_fit | temp | 87.4 | 2.11 | 1.3 | 1.4 | 6.76 | 89.7 | 90.0 |
| refcocog_val | natural_fit | platt | 87.4 | 2.11 | 1.2 | 1.2 | 6.75 | 89.8 | 90.0 |
| refcocog_val | natural_fit | beta | 87.4 | 2.11 | 1.8 | 1.4 | 6.77 | 89.7 | 90.0 |
| refcocog_val | natural_fit | iso | 87.4 | 2.11 | 1.6 | 1.4 | 6.76 | 89.6 | 90.0 |
| refcocog_val | natural_fit | multi | 87.4 | 2.14 | 0.9 | 0.9 | 6.71 | 89.9 | 90.0 |
| refcocog_test | natural_fit | raw | 89.1 | 1.83 | 1.7 | 1.6 | 6.24 | 88.9 | 90.5 |
| refcocog_test | natural_fit | temp | 89.1 | 1.83 | 1.3 | 1.4 | 6.23 | 89.3 | 90.5 |
| refcocog_test | natural_fit | platt | 89.1 | 1.83 | 1.2 | 1.3 | 6.22 | 89.4 | 90.5 |
| refcocog_test | natural_fit | beta | 89.1 | 1.83 | 1.4 | 1.2 | 6.24 | 89.3 | 90.5 |
| refcocog_test | natural_fit | iso | 89.1 | 1.89 | 1.4 | 1.5 | 6.24 | 89.1 | 90.5 |
| refcocog_test | natural_fit | multi | 89.1 | 1.82 | 0.9 | 1.0 | 6.20 | 89.6 | 90.5 |
| grec_val | natural_fit | raw | 90.2 | 1.10 | 0.6 | 0.6 | 4.53 | 92.8 | 93.2 |
| grec_val | natural_fit | temp | 90.2 | 1.10 | 0.6 | 0.5 | 4.53 | 93.1 | 93.2 |
| grec_val | natural_fit | platt | 90.2 | 1.10 | 0.6 | 0.5 | 4.53 | 93.2 | 93.2 |
| grec_val | natural_fit | beta | 90.2 | 1.10 | 0.5 | 0.5 | 4.53 | 93.2 | 93.2 |
| grec_val | natural_fit | iso | 90.2 | 1.10 | 0.7 | 0.8 | 4.54 | 93.2 | 93.2 |
| grec_val | natural_fit | multi | 90.1 | 1.13 | 0.9 | 0.6 | 4.51 | 92.6 | 93.2 |
| grec_testA | natural_fit | raw | 93.5 | 0.90 | 1.4 | 1.2 | 3.98 | 93.2 | 92.5 |
| grec_testA | natural_fit | temp | 93.5 | 0.90 | 1.2 | 1.2 | 3.98 | 93.4 | 92.5 |
| grec_testA | natural_fit | platt | 93.5 | 0.90 | 1.2 | 1.3 | 4.00 | 93.5 | 92.5 |
| grec_testA | natural_fit | beta | 93.5 | 0.90 | 1.2 | 1.3 | 4.02 | 93.6 | 92.5 |
| grec_testA | natural_fit | iso | 93.5 | 0.90 | 1.4 | 1.2 | 4.01 | 93.5 | 92.5 |
| grec_testA | natural_fit | multi | 93.7 | 0.88 | 1.4 | 1.3 | 3.97 | 93.0 | 92.5 |
| grec_testB | natural_fit | raw | 93.1 | 2.90 | 2.3 | 1.9 | 7.15 | 83.9 | 83.2 |
| grec_testB | natural_fit | temp | 93.1 | 2.90 | 2.2 | 1.9 | 7.14 | 84.1 | 83.2 |
| grec_testB | natural_fit | platt | 93.1 | 2.90 | 2.1 | 2.1 | 7.16 | 84.4 | 83.2 |
| grec_testB | natural_fit | beta | 93.1 | 2.90 | 2.9 | 2.5 | 7.31 | 84.7 | 83.2 |
| grec_testB | natural_fit | iso | 93.1 | 2.90 | 2.4 | 1.8 | 7.14 | 84.1 | 83.2 |
| grec_testB | natural_fit | multi | 93.2 | 2.89 | 2.4 | 1.8 | 7.04 | 83.7 | 83.2 |
| calib_natural | natural_fit | raw | 91.1 | 1.45 | 1.2 | 1.2 | 5.88 | 90.1 | 90.5 |
| calib_natural | natural_fit | temp | 91.1 | 1.45 | 1.2 | 1.3 | 5.89 | 90.4 | 90.5 |
| calib_natural | natural_fit | platt | 91.1 | 1.45 | 1.1 | 1.2 | 5.89 | 90.5 | 90.5 |
| calib_natural | natural_fit | beta | 91.1 | 1.45 | 0.9 | 1.0 | 5.88 | 90.5 | 90.5 |
| calib_natural | natural_fit | iso | 91.1 | 1.45 | 0.5 | 0.5 | 5.84 | 90.4 | 90.5 |
| calib_natural | natural_fit | multi | 91.2 | 1.44 | 1.3 | 1.0 | 5.80 | 90.2 | 90.5 |
| guisyn_test | natural_fit | raw | 89.8 | 5.49 | 24.1 | 24.1 | 17.59 | 53.5 | 77.7 |
| guisyn_test | natural_fit | temp | 89.8 | 5.49 | 24.1 | 24.1 | 17.65 | 53.6 | 77.7 |
| guisyn_test | natural_fit | platt | 89.8 | 5.49 | 23.5 | 23.5 | 17.29 | 54.2 | 77.7 |
| guisyn_test | natural_fit | beta | 89.8 | 5.49 | 21.7 | 21.7 | 16.47 | 56.0 | 77.7 |
| guisyn_test | natural_fit | iso | 89.4 | 5.82 | 24.6 | 24.6 | 17.63 | 53.1 | 77.7 |
| guisyn_test | natural_fit | multi | 88.3 | 5.87 | 41.8 | 41.8 | 31.12 | 35.8 | 77.7 |
| guisyn_test | gui_fit | raw | 89.8 | 5.49 | 24.1 | 24.1 | 17.59 | 53.5 | 77.7 |
| guisyn_test | gui_fit | temp | 89.8 | 5.49 | 24.3 | 24.3 | 17.50 | 53.4 | 77.7 |
| guisyn_test | gui_fit | platt | 89.8 | 5.49 | 1.1 | 1.1 | 10.15 | 78.7 | 77.7 |
| guisyn_test | gui_fit | beta | 89.8 | 5.49 | 1.1 | 1.2 | 10.15 | 78.7 | 77.7 |
| guisyn_test | gui_fit | iso | 89.8 | 5.49 | 1.4 | 1.3 | 10.17 | 78.7 | 77.7 |
| guisyn_test | gui_fit | multi | 90.4 | 5.38 | 0.9 | 0.9 | 9.73 | 78.1 | 77.7 |
| ss2 | natural_fit | raw | 82.6 | 3.03 | 40.8 | 40.8 | 28.52 | 48.4 | 89.2 |
| ss2 | natural_fit | temp | 82.6 | 3.03 | 40.9 | 40.9 | 28.82 | 48.3 | 89.2 |
| ss2 | natural_fit | platt | 82.6 | 3.03 | 40.2 | 40.2 | 28.17 | 49.0 | 89.2 |
| ss2 | natural_fit | beta | 82.6 | 3.03 | 38.0 | 38.0 | 25.35 | 51.2 | 89.2 |
| ss2 | natural_fit | iso | 81.5 | 3.39 | 41.2 | 41.2 | 29.11 | 48.1 | 89.2 |
| ss2 | natural_fit | multi | 82.6 | 3.01 | 61.3 | 61.3 | 49.73 | 28.0 | 89.2 |
| ss2 | gui_fit | raw | 82.6 | 3.03 | 40.8 | 40.8 | 28.52 | 48.4 | 89.2 |
| ss2 | gui_fit | temp | 82.6 | 3.03 | 40.7 | 40.7 | 27.91 | 48.5 | 89.2 |
| ss2 | gui_fit | platt | 82.6 | 3.03 | 12.3 | 12.3 | 10.98 | 77.0 | 89.2 |
| ss2 | gui_fit | beta | 82.6 | 3.03 | 12.3 | 12.3 | 10.97 | 77.0 | 89.2 |
| ss2 | gui_fit | iso | 82.6 | 3.03 | 12.3 | 12.3 | 11.08 | 76.9 | 89.2 |
| ss2 | gui_fit | multi | 83.3 | 2.93 | 13.4 | 13.4 | 10.97 | 75.8 | 89.2 |
| venus | natural_fit | raw | 83.7 | 27.45 | 18.5 | 18.5 | 20.98 | 28.8 | 47.3 |
| venus | natural_fit | temp | 83.7 | 27.45 | 18.9 | 18.9 | 21.16 | 28.4 | 47.3 |
| venus | natural_fit | platt | 83.7 | 27.45 | 18.3 | 18.3 | 20.88 | 29.0 | 47.3 |
| venus | natural_fit | beta | 83.7 | 27.45 | 14.9 | 14.9 | 19.47 | 33.8 | 47.3 |
| venus | natural_fit | iso | 80.6 | 30.48 | 19.0 | 19.0 | 21.48 | 28.3 | 47.3 |
| venus | natural_fit | multi | 82.4 | 28.12 | 30.4 | 30.4 | 28.42 | 16.9 | 47.3 |
| venus | gui_fit | raw | 83.7 | 27.45 | 18.5 | 18.5 | 20.98 | 28.8 | 47.3 |
| venus | gui_fit | temp | 83.7 | 27.45 | 17.8 | 17.8 | 20.62 | 29.5 | 47.3 |
| venus | gui_fit | platt | 83.7 | 27.45 | 6.6 | 6.6 | 17.16 | 53.1 | 47.3 |
| venus | gui_fit | beta | 83.7 | 27.45 | 6.6 | 6.6 | 17.17 | 53.0 | 47.3 |
| venus | gui_fit | iso | 83.7 | 27.45 | 6.8 | 6.8 | 17.20 | 53.0 | 47.3 |
| venus | gui_fit | multi | 84.3 | 27.30 | 7.3 | 7.1 | 16.95 | 54.4 | 47.3 |

## Box confidence ĝ·κ̂ vs (present ∧ IoU>=0.5)  [the v0→v1 regression metric]

| set | method | ECE15 | ECE15-mass | Brier | π̂ (EM) |
|---|---|---|---|---|---|
| refcoco_val | raw | 2.43 | 2.42 | 5.35 | - |
| refcoco_val | evalpy_T | 2.32 | 2.32 | 5.35 | - |
| refcoco_val | g:platt|k:platt | 1.75 | 1.93 | 5.22 | - |
| refcoco_val | g:platt+em|k:platt | 0.57 | 0.41 | 4.55 | 0.999 |
| refcoco_val | g:platt|k:iso | 1.77 | 2.10 | 5.24 | - |
| refcoco_val | g:platt+em|k:iso | 0.69 | 0.73 | 4.56 | 0.999 |
| refcoco_val | g:platt|k:multi | 2.32 | 2.32 | 5.21 | - |
| refcoco_val | g:platt+em|k:multi | 0.97 | 0.64 | 4.53 | 0.999 |
| refcoco_val | g:iso|k:platt | 1.74 | 1.91 | 5.25 | - |
| refcoco_val | g:iso+em|k:platt | 0.56 | 0.40 | 4.55 | 0.999 |
| refcoco_val | g:iso|k:iso | 1.84 | 2.11 | 5.27 | - |
| refcoco_val | g:iso+em|k:iso | 0.68 | 0.74 | 4.56 | 0.999 |
| refcoco_val | g:iso|k:multi | 2.34 | 2.34 | 5.24 | - |
| refcoco_val | g:iso+em|k:multi | 0.95 | 0.64 | 4.53 | 0.999 |
| refcoco_val | kappa_only_raw | 0.60 | 0.50 | 4.54 | - |
| refcoco_testA | raw | 1.80 | 1.88 | 4.21 | - |
| refcoco_testA | evalpy_T | 1.69 | 1.77 | 4.20 | - |
| refcoco_testA | g:platt|k:platt | 1.21 | 1.36 | 4.11 | - |
| refcoco_testA | g:platt+em|k:platt | 0.70 | 0.68 | 3.59 | 0.999 |
| refcoco_testA | g:platt|k:iso | 1.15 | 1.47 | 4.13 | - |
| refcoco_testA | g:platt+em|k:iso | 0.87 | 0.90 | 3.61 | 0.999 |
| refcoco_testA | g:platt|k:multi | 1.79 | 1.76 | 4.09 | - |
| refcoco_testA | g:platt+em|k:multi | 0.97 | 0.90 | 3.56 | 0.999 |
| refcoco_testA | g:iso|k:platt | 1.28 | 1.32 | 4.14 | - |
| refcoco_testA | g:iso+em|k:platt | 0.66 | 0.72 | 3.58 | 0.999 |
| refcoco_testA | g:iso|k:iso | 1.21 | 1.37 | 4.17 | - |
| refcoco_testA | g:iso+em|k:iso | 0.87 | 0.91 | 3.61 | 0.999 |
| refcoco_testA | g:iso|k:multi | 1.92 | 1.75 | 4.12 | - |
| refcoco_testA | g:iso+em|k:multi | 0.97 | 0.91 | 3.55 | 0.999 |
| refcoco_testA | kappa_only_raw | 0.74 | 0.72 | 3.58 | - |
| refcoco_testB | raw | 2.99 | 3.24 | 7.19 | - |
| refcoco_testB | evalpy_T | 2.89 | 3.23 | 7.19 | - |
| refcoco_testB | g:platt|k:platt | 2.77 | 2.78 | 7.04 | - |
| refcoco_testB | g:platt+em|k:platt | 1.55 | 1.51 | 6.26 | 0.999 |
| refcoco_testB | g:platt|k:iso | 2.68 | 2.70 | 7.07 | - |
| refcoco_testB | g:platt+em|k:iso | 1.28 | 1.49 | 6.28 | 0.999 |
| refcoco_testB | g:platt|k:multi | 2.88 | 2.94 | 7.04 | - |
| refcoco_testB | g:platt+em|k:multi | 1.43 | 1.29 | 6.24 | 0.999 |
| refcoco_testB | g:iso|k:platt | 2.74 | 2.76 | 7.09 | - |
| refcoco_testB | g:iso+em|k:platt | 1.56 | 1.50 | 6.26 | 0.999 |
| refcoco_testB | g:iso|k:iso | 2.73 | 2.74 | 7.11 | - |
| refcoco_testB | g:iso+em|k:iso | 1.25 | 1.45 | 6.27 | 0.999 |
| refcoco_testB | g:iso|k:multi | 2.90 | 2.96 | 7.08 | - |
| refcoco_testB | g:iso+em|k:multi | 1.43 | 1.29 | 6.24 | 0.999 |
| refcoco_testB | kappa_only_raw | 1.25 | 1.51 | 6.26 | - |
| refcocoplus_val | raw | 3.46 | 3.46 | 7.17 | - |
| refcocoplus_val | evalpy_T | 3.35 | 3.35 | 7.16 | - |
| refcocoplus_val | g:platt|k:platt | 2.68 | 2.71 | 7.00 | - |
| refcocoplus_val | g:platt+em|k:platt | 1.06 | 1.00 | 6.25 | 0.999 |
| refcocoplus_val | g:platt|k:iso | 2.79 | 2.97 | 7.02 | - |
| refcocoplus_val | g:platt+em|k:iso | 0.97 | 1.05 | 6.26 | 0.999 |
| refcocoplus_val | g:platt|k:multi | 3.10 | 3.10 | 6.98 | - |
| refcocoplus_val | g:platt+em|k:multi | 1.20 | 1.14 | 6.22 | 0.999 |
| refcocoplus_val | g:iso|k:platt | 2.67 | 2.67 | 7.02 | - |
| refcocoplus_val | g:iso+em|k:platt | 1.05 | 1.02 | 6.25 | 0.999 |
| refcocoplus_val | g:iso|k:iso | 2.78 | 2.91 | 7.04 | - |
| refcocoplus_val | g:iso+em|k:iso | 0.99 | 1.04 | 6.25 | 0.999 |
| refcocoplus_val | g:iso|k:multi | 3.09 | 3.09 | 7.00 | - |
| refcocoplus_val | g:iso+em|k:multi | 1.19 | 1.14 | 6.22 | 0.999 |
| refcocoplus_val | kappa_only_raw | 1.29 | 1.14 | 6.26 | - |
| refcocoplus_testA | raw | 2.75 | 2.89 | 5.24 | - |
| refcocoplus_testA | evalpy_T | 2.65 | 2.72 | 5.23 | - |
| refcocoplus_testA | g:platt|k:platt | 2.10 | 2.26 | 5.11 | - |
| refcocoplus_testA | g:platt+em|k:platt | 0.81 | 0.99 | 4.44 | 0.999 |
| refcocoplus_testA | g:platt|k:iso | 2.07 | 2.39 | 5.10 | - |
| refcocoplus_testA | g:platt+em|k:iso | 0.94 | 0.91 | 4.43 | 0.999 |
| refcocoplus_testA | g:platt|k:multi | 2.64 | 2.79 | 5.11 | - |
| refcocoplus_testA | g:platt+em|k:multi | 1.09 | 1.15 | 4.43 | 0.999 |
| refcocoplus_testA | g:iso|k:platt | 2.11 | 2.24 | 5.13 | - |
| refcocoplus_testA | g:iso+em|k:platt | 0.81 | 0.99 | 4.44 | 0.999 |
| refcocoplus_testA | g:iso|k:iso | 2.06 | 2.35 | 5.13 | - |
| refcocoplus_testA | g:iso+em|k:iso | 0.92 | 0.91 | 4.43 | 0.999 |
| refcocoplus_testA | g:iso|k:multi | 2.65 | 2.78 | 5.14 | - |
| refcocoplus_testA | g:iso+em|k:multi | 1.09 | 1.15 | 4.43 | 0.999 |
| refcocoplus_testA | kappa_only_raw | 1.06 | 1.08 | 4.44 | - |
| refcocoplus_testB | raw | 4.10 | 4.32 | 9.78 | - |
| refcocoplus_testB | evalpy_T | 3.98 | 4.18 | 9.77 | - |
| refcocoplus_testB | g:platt|k:platt | 3.18 | 3.53 | 9.61 | - |
| refcocoplus_testB | g:platt+em|k:platt | 1.23 | 1.47 | 8.87 | 0.999 |
| refcocoplus_testB | g:platt|k:iso | 3.43 | 3.64 | 9.65 | - |
| refcocoplus_testB | g:platt+em|k:iso | 1.49 | 1.65 | 8.89 | 0.999 |
| refcocoplus_testB | g:platt|k:multi | 3.48 | 3.53 | 9.60 | - |
| refcocoplus_testB | g:platt+em|k:multi | 1.43 | 1.49 | 8.86 | 0.999 |
| refcocoplus_testB | g:iso|k:platt | 3.24 | 3.48 | 9.65 | - |
| refcocoplus_testB | g:iso+em|k:platt | 1.23 | 1.47 | 8.87 | 0.999 |
| refcocoplus_testB | g:iso|k:iso | 3.43 | 3.55 | 9.69 | - |
| refcocoplus_testB | g:iso+em|k:iso | 1.50 | 1.64 | 8.89 | 0.999 |
| refcocoplus_testB | g:iso|k:multi | 3.48 | 3.61 | 9.64 | - |
| refcocoplus_testB | g:iso+em|k:multi | 1.43 | 1.49 | 8.86 | 0.999 |
| refcocoplus_testB | kappa_only_raw | 1.65 | 1.76 | 8.88 | - |
| refcocog_val | raw | 4.42 | 4.63 | 8.00 | - |
| refcocog_val | evalpy_T | 4.30 | 4.39 | 7.98 | - |
| refcocog_val | g:platt|k:platt | 3.34 | 3.82 | 7.76 | - |
| refcocog_val | g:platt+em|k:platt | 1.13 | 1.23 | 6.76 | 0.999 |
| refcocog_val | g:platt|k:iso | 3.56 | 3.98 | 7.79 | - |
| refcocog_val | g:platt+em|k:iso | 1.54 | 1.50 | 6.77 | 0.999 |
| refcocog_val | g:platt|k:multi | 3.32 | 3.59 | 7.70 | - |
| refcocog_val | g:platt+em|k:multi | 0.90 | 0.96 | 6.72 | 0.999 |
| refcocog_val | g:iso|k:platt | 3.32 | 3.73 | 7.77 | - |
| refcocog_val | g:iso+em|k:platt | 1.19 | 1.23 | 6.75 | 0.999 |
| refcocog_val | g:iso|k:iso | 3.54 | 3.81 | 7.80 | - |
| refcocog_val | g:iso+em|k:iso | 1.52 | 1.42 | 6.77 | 0.999 |
| refcocog_val | g:iso|k:multi | 3.29 | 3.52 | 7.72 | - |
| refcocog_val | g:iso+em|k:multi | 0.84 | 0.96 | 6.72 | 0.999 |
| refcocog_val | kappa_only_raw | 1.18 | 1.38 | 6.76 | - |
| refcocog_test | raw | 5.02 | 5.02 | 7.55 | - |
| refcocog_test | evalpy_T | 4.87 | 4.88 | 7.54 | - |
| refcocog_test | g:platt|k:platt | 4.03 | 4.04 | 7.29 | - |
| refcocog_test | g:platt+em|k:platt | 1.26 | 1.37 | 6.23 | 0.999 |
| refcocog_test | g:platt|k:iso | 4.31 | 4.40 | 7.34 | - |
| refcocog_test | g:platt+em|k:iso | 1.45 | 1.54 | 6.25 | 0.999 |
| refcocog_test | g:platt|k:multi | 3.93 | 3.96 | 7.25 | - |
| refcocog_test | g:platt+em|k:multi | 0.94 | 1.04 | 6.21 | 0.999 |
| refcocog_test | g:iso|k:platt | 4.13 | 4.02 | 7.34 | - |
| refcocog_test | g:iso+em|k:platt | 1.25 | 1.36 | 6.23 | 0.999 |
| refcocog_test | g:iso|k:iso | 4.29 | 4.37 | 7.38 | - |
| refcocog_test | g:iso+em|k:iso | 1.47 | 1.53 | 6.25 | 0.999 |
| refcocog_test | g:iso|k:multi | 3.92 | 3.92 | 7.30 | - |
| refcocog_test | g:iso+em|k:multi | 0.96 | 1.04 | 6.21 | 0.999 |
| refcocog_test | kappa_only_raw | 1.71 | 1.65 | 6.24 | - |
| grec_val | raw | 3.76 | 3.77 | 5.89 | - |
| grec_val | evalpy_T | 3.78 | 3.79 | 5.90 | - |
| grec_val | g:platt|k:platt | 4.55 | 4.56 | 6.12 | - |
| grec_val | g:platt+em|k:platt | 3.19 | 3.20 | 5.67 | 0.59 |
| grec_val | g:platt|k:iso | 4.50 | 4.51 | 6.11 | - |
| grec_val | g:platt+em|k:iso | 3.15 | 3.16 | 5.67 | 0.59 |
| grec_val | g:platt|k:multi | 4.19 | 4.20 | 6.09 | - |
| grec_val | g:platt+em|k:multi | 2.84 | 2.85 | 5.65 | 0.59 |
| grec_val | g:iso|k:platt | 4.41 | 4.42 | 6.10 | - |
| grec_val | g:iso+em|k:platt | 3.10 | 3.12 | 5.71 | 0.589 |
| grec_val | g:iso|k:iso | 4.37 | 4.38 | 6.08 | - |
| grec_val | g:iso+em|k:iso | 3.06 | 3.07 | 5.70 | 0.589 |
| grec_val | g:iso|k:multi | 4.05 | 4.06 | 6.07 | - |
| grec_val | g:iso+em|k:multi | 2.75 | 2.76 | 5.68 | 0.589 |
| grec_val | kappa_only_raw | 23.35 | 23.35 | 17.97 | - |
| grec_testA | raw | 6.27 | 6.46 | 7.53 | - |
| grec_testA | evalpy_T | 6.32 | 6.50 | 7.55 | - |
| grec_testA | g:platt|k:platt | 7.04 | 7.20 | 7.83 | - |
| grec_testA | g:platt+em|k:platt | 6.11 | 6.25 | 7.40 | 0.626 |
| grec_testA | g:platt|k:iso | 7.01 | 7.21 | 7.81 | - |
| grec_testA | g:platt+em|k:iso | 6.12 | 6.26 | 7.38 | 0.626 |
| grec_testA | g:platt|k:multi | 6.72 | 6.87 | 7.79 | - |
| grec_testA | g:platt+em|k:multi | 5.80 | 5.92 | 7.36 | 0.626 |
| grec_testA | g:iso|k:platt | 6.87 | 7.04 | 7.83 | - |
| grec_testA | g:iso+em|k:platt | 5.87 | 6.15 | 7.44 | 0.625 |
| grec_testA | g:iso|k:iso | 6.87 | 7.04 | 7.81 | - |
| grec_testA | g:iso+em|k:iso | 5.93 | 6.16 | 7.42 | 0.625 |
| grec_testA | g:iso|k:multi | 6.55 | 6.71 | 7.78 | - |
| grec_testA | g:iso+em|k:multi | 5.57 | 5.82 | 7.40 | 0.625 |
| grec_testA | kappa_only_raw | 25.00 | 25.00 | 19.83 | - |
| grec_testB | raw | 8.13 | 8.20 | 10.81 | - |
| grec_testB | evalpy_T | 8.18 | 8.27 | 10.85 | - |
| grec_testB | g:platt|k:platt | 9.01 | 9.03 | 11.16 | - |
| grec_testB | g:platt+em|k:platt | 7.50 | 7.58 | 10.49 | 0.575 |
| grec_testB | g:platt|k:iso | 8.83 | 8.82 | 11.10 | - |
| grec_testB | g:platt+em|k:iso | 7.34 | 7.43 | 10.44 | 0.575 |
| grec_testB | g:platt|k:multi | 8.64 | 8.67 | 11.08 | - |
| grec_testB | g:platt+em|k:multi | 7.15 | 7.21 | 10.42 | 0.575 |
| grec_testB | g:iso|k:platt | 8.80 | 8.80 | 11.17 | - |
| grec_testB | g:iso+em|k:platt | 7.38 | 7.48 | 10.57 | 0.575 |
| grec_testB | g:iso|k:iso | 8.62 | 8.64 | 11.10 | - |
| grec_testB | g:iso+em|k:iso | 7.22 | 7.32 | 10.52 | 0.575 |
| grec_testB | g:iso|k:multi | 8.44 | 8.45 | 11.09 | - |
| grec_testB | g:iso+em|k:multi | 7.04 | 7.13 | 10.50 | 0.575 |
| grec_testB | kappa_only_raw | 26.21 | 26.21 | 22.17 | - |

Bootstrap (image clusters): 
- refcoco_val: raw ECE 2.43 [1.98, 2.99]; fixed 0.57 [0.50, 1.17]; paired diff -1.86 [-2.18, -1.06]; κ̂ AUROC 90.1 [88.6, 91.6]
- refcocog_test: raw ECE 5.02 [4.48, 5.64]; fixed 1.26 [0.97, 1.91]; paired diff -3.77 [-3.99, -3.12]; κ̂ AUROC 89.1 [88.0, 90.2]

## Presence bootstrap CIs (image clusters, raw ĝ unless noted)

- grec_val: AUROC 97.9 [97.7, 98.1]; N-acc 87.4 [86.6, 88.0]; ECE raw 2.27 [2.01, 2.59] → Platt 3.02 [2.77, 3.33] (paired 0.75 [0.65, 0.82]); subset ECE raw 5.87 [5.41, 6.37] → Platt 7.01 [6.55, 7.50]
- grec_testA: AUROC 94.1 [93.5, 94.6]; N-acc 80.8 [79.6, 82.0]; ECE raw 4.43 [4.00, 5.00] → Platt 4.23 [3.85, 4.81] (paired -0.19 [-0.36, 0.05]); subset ECE raw 4.43 [4.00, 5.00] → Platt 4.23 [3.85, 4.81]
- grec_testB: AUROC 88.7 [87.5, 90.1]; N-acc 77.1 [75.9, 78.4]; ECE raw 8.45 [7.41, 9.61] → Platt 8.87 [7.79, 10.02] (paired 0.43 [0.19, 0.55]); subset ECE raw 8.45 [7.41, 9.61] → Platt 8.87 [7.79, 10.02]
- venus: AUROC 73.8 [71.9, 75.5]; N-acc 81.8 [79.2, 84.5]; ECE raw 37.85 [36.22, 39.40] → Platt 21.53 [20.36, 22.56] (paired -16.32 [-17.11, -15.60]); subset ECE raw 37.85 [36.22, 39.40] → Platt 21.53 [20.36, 22.56]

## Risk-controlled accept/abstain (δ=0.1; 50/50 image splits ×reps; coverage = fraction answered)


**refcoco_val** (n=10834, absent 0.0%, correct-if-answered-all 93.2%); default ĝ≥.5: coverage 98.6, selective risk 6.7, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.2 | 0.03 | 0.015 |
| selective[g] | 0.02 | 17.9 | 0.84 | 0.075 |
| selective[g] | 0.05 | 83.4 | 4.17 | 0.09 |
| selective[g] | 0.1 | 100.0 | 6.80 | 0.0 |
| joint[g] | 0.01 | 46.3 | 0.85 | 0.31 |
| joint[g] | 0.02 | 65.7 | 1.77 | 0.25 |
| joint[g] | 0.05 | 89.3 | 4.52 | 0.18 |
| joint[g] | 0.1 | 100.0 | 6.81 | 0.0 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 16.3 | 0.91 | False |
| selective_transfer[g] | 0.05 | 71.2 | 3.03 | False |
| selective_transfer[g] | 0.1 | 95.2 | 6.14 | False |
| selective[g*k] | 0.01 | 17.0 | 0.33 | 0.115 |
| selective[g*k] | 0.02 | 76.2 | 1.52 | 0.14 |
| selective[g*k] | 0.05 | 93.8 | 4.19 | 0.115 |
| selective[g*k] | 0.1 | 100.0 | 6.80 | 0.0 |
| joint[g*k] | 0.01 | 69.3 | 0.82 | 0.225 |
| joint[g*k] | 0.02 | 82.8 | 1.74 | 0.25 |
| joint[g*k] | 0.05 | 95.3 | 4.48 | 0.17 |
| joint[g*k] | 0.1 | 100.0 | 6.81 | 0.0 |
| selective_transfer[g*k] | 0.01 | 37.3 | 0.59 | False |
| selective_transfer[g*k] | 0.02 | 67.7 | 1.09 | False |
| selective_transfer[g*k] | 0.05 | 88.1 | 2.84 | False |
| selective_transfer[g*k] | 0.1 | 97.2 | 5.49 | False |

AURC ĝ 2.44 vs ĝκ̂ 1.26; paired diff -1.19 [-1.47, -0.90]

**refcocog_test** (n=9602, absent 0.0%, correct-if-answered-all 90.5%); default ĝ≥.5: coverage 97.4, selective risk 9.0, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 3.6 | 0.26 | 0.03 |
| selective[g] | 0.05 | 61.0 | 4.09 | 0.08 |
| selective[g] | 0.1 | 96.1 | 8.86 | 0.1 |
| joint[g] | 0.01 | 36.4 | 0.81 | 0.18 |
| joint[g] | 0.02 | 53.6 | 1.75 | 0.215 |
| joint[g] | 0.05 | 75.6 | 4.57 | 0.15 |
| joint[g] | 0.1 | 99.2 | 9.25 | 0.11 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 4.3 | 0.96 | False |
| selective_transfer[g] | 0.05 | 47.3 | 2.86 | False |
| selective_transfer[g] | 0.1 | 90.8 | 8.18 | False |
| selective[g*k] | 0.01 | 9.7 | 0.24 | 0.035 |
| selective[g*k] | 0.02 | 61.7 | 1.40 | 0.05 |
| selective[g*k] | 0.05 | 83.8 | 4.21 | 0.035 |
| selective[g*k] | 0.1 | 98.7 | 8.88 | 0.09 |
| joint[g*k] | 0.01 | 60.7 | 0.81 | 0.17 |
| joint[g*k] | 0.02 | 72.8 | 1.71 | 0.17 |
| joint[g*k] | 0.05 | 88.0 | 4.56 | 0.12 |
| joint[g*k] | 0.1 | 99.5 | 9.26 | 0.11 |
| selective_transfer[g*k] | 0.01 | 27.8 | 0.56 | False |
| selective_transfer[g*k] | 0.02 | 53.6 | 0.95 | False |
| selective_transfer[g*k] | 0.05 | 81.0 | 3.56 | False |
| selective_transfer[g*k] | 0.1 | 95.2 | 7.32 | False |

AURC ĝ 3.82 vs ĝκ̂ 2.01; paired diff -1.81 [-2.11, -1.50]

**grec_val** (n=25063, absent 35.5%, correct-if-answered-all 56.0%); default ĝ≥.5: coverage 67.3, selective risk 18.6, false-accept on absent 12.6

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.2 | 0.17 | 0.025 |
| selective[g] | 0.1 | 39.8 | 8.97 | 0.08 |
| joint[g] | 0.01 | 17.2 | 0.89 | 0.23 |
| joint[g] | 0.02 | 27.9 | 1.82 | 0.18 |
| joint[g] | 0.05 | 45.9 | 4.74 | 0.225 |
| joint[g] | 0.1 | 62.0 | 9.54 | 0.13 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 7.7 | 4.90 | True |
| selective_transfer[g] | 0.05 | 37.6 | 8.43 | True |
| selective_transfer[g] | 0.1 | 60.5 | 14.70 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 2.0 | 0.31 | 0.015 |
| selective[g*k] | 0.05 | 39.5 | 4.24 | 0.045 |
| selective[g*k] | 0.1 | 53.2 | 9.11 | 0.07 |
| joint[g*k] | 0.01 | 30.9 | 0.88 | 0.2 |
| joint[g*k] | 0.02 | 40.6 | 1.83 | 0.165 |
| joint[g*k] | 0.05 | 52.7 | 4.66 | 0.135 |
| joint[g*k] | 0.1 | 63.0 | 9.52 | 0.17 |
| selective_transfer[g*k] | 0.01 | 16.7 | 1.81 | True |
| selective_transfer[g*k] | 0.02 | 32.4 | 3.09 | True |
| selective_transfer[g*k] | 0.05 | 48.9 | 7.18 | True |
| selective_transfer[g*k] | 0.1 | 64.6 | 16.30 | True |
| absent_conformal[g] | 0.01 | present accepted 56.3 | false-accept absent 1.01 | |
| absent_conformal[g] | 0.02 | present accepted 73.6 | false-accept absent 1.98 | |
| absent_conformal[g] | 0.05 | present accepted 90.1 | false-accept absent 4.99 | |
| absent_conformal[g] | 0.1 | present accepted 96.1 | false-accept absent 10.04 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 83.0 | false-accept absent 3.14 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 92.1 | false-accept absent 6.02 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 97.3 | false-accept absent 12.45 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 99.0 | false-accept absent 21.45 | |

AURC ĝ 16.53 vs ĝκ̂ 14.19; paired diff -2.34 [-2.76, -2.03]

**grec_testA** (n=19200, absent 23.2%, correct-if-answered-all 60.1%); default ĝ≥.5: coverage 77.0, selective risk 24.4, false-accept on absent 19.2

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 5.5 | 0.88 | 0.34 |
| joint[g] | 0.02 | 12.6 | 1.80 | 0.28 |
| joint[g] | 0.05 | 29.7 | 4.79 | 0.35 |
| joint[g] | 0.1 | 51.2 | 9.53 | 0.22 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 8.5 | 13.94 | True |
| selective_transfer[g] | 0.05 | 40.7 | 17.15 | True |
| selective_transfer[g] | 0.1 | 69.1 | 21.82 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 25.8 | 3.97 | 0.06 |
| selective[g*k] | 0.1 | 45.8 | 8.91 | 0.09 |
| joint[g*k] | 0.01 | 24.1 | 0.86 | 0.165 |
| joint[g*k] | 0.02 | 33.8 | 1.81 | 0.25 |
| joint[g*k] | 0.05 | 47.5 | 4.64 | 0.21 |
| joint[g*k] | 0.1 | 60.1 | 9.44 | 0.18 |
| selective_transfer[g*k] | 0.01 | 16.0 | 2.79 | True |
| selective_transfer[g*k] | 0.02 | 29.2 | 4.50 | True |
| selective_transfer[g*k] | 0.05 | 49.1 | 10.46 | True |
| selective_transfer[g*k] | 0.1 | 73.2 | 21.89 | True |
| absent_conformal[g] | 0.01 | present accepted 22.8 | false-accept absent 0.97 | |
| absent_conformal[g] | 0.02 | present accepted 35.6 | false-accept absent 1.97 | |
| absent_conformal[g] | 0.05 | present accepted 67.9 | false-accept absent 5.02 | |
| absent_conformal[g] | 0.1 | present accepted 84.4 | false-accept absent 9.91 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 77.8 | false-accept absent 7.15 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 87.9 | false-accept absent 11.83 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 94.3 | false-accept absent 19.09 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 97.1 | false-accept absent 27.27 | |

AURC ĝ 21.30 vs ĝκ̂ 14.35; paired diff -6.96 [-7.74, -5.99]

**grec_testB** (n=16063, absent 29.1%, correct-if-answered-all 51.0%); default ĝ≥.5: coverage 70.6, selective risk 31.2, false-accept on absent 22.9

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.1 | 0.20 | 0.015 |
| joint[g] | 0.01 | 6.8 | 0.84 | 0.2 |
| joint[g] | 0.02 | 12.5 | 1.80 | 0.215 |
| joint[g] | 0.05 | 25.5 | 4.65 | 0.245 |
| joint[g] | 0.1 | 41.7 | 9.49 | 0.205 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 4.5 | 10.90 | True |
| selective_transfer[g] | 0.05 | 30.1 | 19.35 | True |
| selective_transfer[g] | 0.1 | 62.2 | 28.30 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 22.1 | 8.33 | 0.08 |
| joint[g*k] | 0.01 | 13.9 | 0.87 | 0.255 |
| joint[g*k] | 0.02 | 21.8 | 1.78 | 0.225 |
| joint[g*k] | 0.05 | 35.1 | 4.64 | 0.19 |
| joint[g*k] | 0.1 | 48.3 | 9.35 | 0.11 |
| selective_transfer[g*k] | 0.01 | 8.6 | 5.06 | True |
| selective_transfer[g*k] | 0.02 | 21.0 | 7.94 | True |
| selective_transfer[g*k] | 0.05 | 40.3 | 15.57 | True |
| selective_transfer[g*k] | 0.1 | 64.6 | 27.82 | True |
| absent_conformal[g] | 0.01 | present accepted 9.1 | false-accept absent 1.00 | |
| absent_conformal[g] | 0.02 | present accepted 21.6 | false-accept absent 1.95 | |
| absent_conformal[g] | 0.05 | present accepted 47.1 | false-accept absent 4.94 | |
| absent_conformal[g] | 0.1 | present accepted 71.5 | false-accept absent 10.01 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 71.4 | false-accept absent 9.91 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 83.7 | false-accept absent 14.92 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 90.1 | false-accept absent 22.75 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 92.8 | false-accept absent 31.95 | |

AURC ĝ 26.36 vs ĝκ̂ 22.34; paired diff -4.02 [-4.66, -3.47]

**ss2** (n=1272, absent 0.0%, correct-if-answered-all 89.2%); default ĝ≥.5: coverage 83.6, selective risk 7.9, false-accept on absent -

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 4.0 | 0.43 | 0.015 |
| selective[g] | 0.1 | 78.8 | 6.90 | 0.035 |
| joint[g] | 0.01 | 14.8 | 0.44 | 0.07 |
| joint[g] | 0.02 | 38.0 | 1.27 | 0.13 |
| joint[g] | 0.05 | 70.7 | 4.00 | 0.18 |
| joint[g] | 0.1 | 93.6 | 8.44 | 0.16 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.1 | 0.4 | 0.00 | False |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 50.4 | 2.52 | 0.05 |
| selective[g*k] | 0.1 | 91.3 | 7.15 | 0.03 |
| joint[g*k] | 0.01 | 36.6 | 0.44 | 0.075 |
| joint[g*k] | 0.02 | 53.2 | 1.24 | 0.09 |
| joint[g*k] | 0.05 | 78.8 | 3.99 | 0.18 |
| joint[g*k] | 0.1 | 96.5 | 8.41 | 0.175 |
| selective_transfer[g*k] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.05 | 52.3 | 2.26 | False |
| selective_transfer[g*k] | 0.1 | 77.4 | 4.57 | False |

AURC ĝ 4.82 vs ĝκ̂ 2.91; paired diff -1.91 [-2.94, -1.05]

**venus** (n=6165, absent 14.6%, correct-if-answered-all 40.4%); default ĝ≥.5: coverage 48.3, selective risk 41.8, false-accept on absent 18.2

| rule | α | coverage | realized risk | violation rate |
|---|---|---|---|---|
| selective[g] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g] | 0.1 | 0.0 | 0.00 | 0.0 |
| joint[g] | 0.01 | 4.8 | 0.79 | 0.12 |
| joint[g] | 0.02 | 8.1 | 1.67 | 0.2 |
| joint[g] | 0.05 | 16.7 | 4.51 | 0.19 |
| joint[g] | 0.1 | 28.3 | 9.11 | 0.205 |
| selective_transfer[g] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.05 | 0.0 | 0.00 | False |
| selective_transfer[g] | 0.1 | 0.4 | 13.64 | True |
| selective[g*k] | 0.01 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.02 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.05 | 0.0 | 0.00 | 0.0 |
| selective[g*k] | 0.1 | 0.2 | 0.79 | 0.025 |
| joint[g*k] | 0.01 | 6.5 | 0.78 | 0.145 |
| joint[g*k] | 0.02 | 10.9 | 1.61 | 0.135 |
| joint[g*k] | 0.05 | 21.5 | 4.49 | 0.195 |
| joint[g*k] | 0.1 | 32.6 | 9.07 | 0.155 |
| selective_transfer[g*k] | 0.01 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.02 | 0.0 | 0.00 | False |
| selective_transfer[g*k] | 0.05 | 23.9 | 22.43 | True |
| selective_transfer[g*k] | 0.1 | 43.6 | 34.28 | True |
| absent_conformal[g] | 0.01 | present accepted 7.9 | false-accept absent 0.91 | |
| absent_conformal[g] | 0.02 | present accepted 12.6 | false-accept absent 1.99 | |
| absent_conformal[g] | 0.05 | present accepted 26.2 | false-accept absent 4.90 | |
| absent_conformal[g] | 0.1 | present accepted 41.4 | false-accept absent 9.83 | |
| absent_conformal_transfer[g] | 0.01 | present accepted 0.2 | false-accept absent 0.00 | |
| absent_conformal_transfer[g] | 0.02 | present accepted 0.6 | false-accept absent 0.00 | |
| absent_conformal_transfer[g] | 0.05 | present accepted 3.1 | false-accept absent 0.11 | |
| absent_conformal_transfer[g] | 0.1 | present accepted 8.6 | false-accept absent 0.89 | |

AURC ĝ 40.63 vs ĝκ̂ 35.68; paired diff -4.95 [-5.75, -4.21]

## Prop. 1 two-knob plug-in rule vs one-knob thresholds (expected cost per query, lower is better)

| set | costs | two-knob plug-in | ĝ≥.5 | ĝ thr tuned on calib | ĝκ̂ thr tuned | ĝ oracle thr | paired two−one(tuned) |
|---|---|---|---|---|---|---|---|
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.0719 | 0.0724 | 0.0750 | 0.0678 | 0.0682 | -0.0030 [-0.0048, -0.0011] |
| refcoco_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1082 | 0.1380 | 0.1421 | 0.1075 | 0.1356 | -0.0339 [-0.0412, -0.0278] |
| refcoco_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.0785 | 0.0792 | 0.0819 | 0.0759 | 0.0689 | -0.0033 [-0.0047, -0.0023] |
| refcoco_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.0764 | 0.0724 | 0.0818 | 0.0720 | 0.0682 | -0.0054 [-0.0073, -0.0034] |
| refcoco_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1512 | 0.2693 | 0.2499 | 0.1559 | 0.2499 | -0.0986 [-0.1152, -0.0824] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1043 | 0.1006 | 0.1073 | 0.0959 | 0.0956 | -0.0030 [-0.0052, -0.0007] |
| refcocog_test | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1497 | 0.1879 | 0.1972 | 0.1514 | 0.1875 | -0.0475 [-0.0550, -0.0410] |
| refcocog_test | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1148 | 0.1137 | 0.1177 | 0.1102 | 0.0964 | -0.0029 [-0.0042, -0.0013] |
| refcocog_test | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1117 | 0.1006 | 0.1179 | 0.1040 | 0.0956 | -0.0062 [-0.0084, -0.0038] |
| refcocog_test | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1979 | 0.3627 | 0.3282 | 0.2105 | 0.3282 | -0.1303 [-0.1495, -0.1134] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1130 | 0.1337 | 0.1243 | 0.1224 | 0.1182 | -0.0114 [-0.0135, -0.0093] |
| grec_val | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.1403 | 0.2141 | 0.1865 | 0.1401 | 0.1826 | -0.0463 [-0.0524, -0.0405] |
| grec_val | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1412 | 0.1421 | 0.1404 | 0.1502 | 0.1382 | 0.0008 [-0.0005, 0.0021] |
| grec_val | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.1325 | 0.1786 | 0.1388 | 0.1449 | 0.1339 | -0.0063 [-0.0086, -0.0035] |
| grec_val | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.1662 | 0.3749 | 0.3086 | 0.1661 | 0.3086 | -0.1425 [-0.1580, -0.1277] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.1919 | 0.2092 | 0.2044 | 0.1970 | 0.2022 | -0.0125 [-0.0163, -0.0093] |
| grec_testA | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2314 | 0.3523 | 0.3252 | 0.2373 | 0.3216 | -0.0938 [-0.1034, -0.0835] |
| grec_testA | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2324 | 0.2308 | 0.2317 | 0.2361 | 0.2303 | 0.0008 [-0.0007, 0.0022] |
| grec_testA | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2176 | 0.2538 | 0.2291 | 0.2248 | 0.2261 | -0.0115 [-0.0150, -0.0077] |
| grec_testA | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2475 | 0.6385 | 0.5536 | 0.2486 | 0.5536 | -0.3061 [-0.3292, -0.2798] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2282 | 0.2553 | 0.2463 | 0.2385 | 0.2385 | -0.0181 [-0.0228, -0.0136] |
| grec_testB | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2633 | 0.4090 | 0.3700 | 0.2676 | 0.3605 | -0.1067 [-0.1185, -0.0944] |
| grec_testB | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.2916 | 0.2903 | 0.2886 | 0.3071 | 0.2874 | 0.0031 [0.0013, 0.0046] |
| grec_testB | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2661 | 0.3219 | 0.2825 | 0.2792 | 0.2718 | -0.0164 [-0.0211, -0.0117] |
| grec_testB | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.2823 | 0.7164 | 0.6047 | 0.2738 | 0.6047 | -0.3224 [-0.3487, -0.2944] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.2103 | 0.1478 | 0.1608 | 0.1612 | 0.1061 | 0.0495 [0.0379, 0.0609] |
| ss2 | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.2759 | 0.2138 | 0.2300 | 0.2300 | 0.1954 | 0.0460 [0.0267, 0.0657] |
| ss2 | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.1847 | 0.2296 | 0.1903 | 0.1950 | 0.1077 | -0.0055 [-0.0165, 0.0054] |
| ss2 | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.2728 | 0.1478 | 0.1918 | 0.1741 | 0.1061 | 0.0810 [0.0688, 0.0951] |
| ss2 | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.3325 | 0.3459 | 0.3412 | 0.2826 | 0.2901 | -0.0086 [-0.0411, 0.0191] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 0.5} | 0.3659 | 0.4006 | 0.3881 | 0.3780 | 0.3792 | -0.0221 [-0.0286, -0.0162] |
| venus | {"c_H": 1, "c_W": 2, "c_R": 0.5} | 0.4037 | 0.5760 | 0.4674 | 0.4220 | 0.4182 | -0.0637 [-0.0790, -0.0500] |
| venus | {"c_H": 1, "c_W": 1, "c_R": 1.0} | 0.5977 | 0.5994 | 0.5862 | 0.6023 | 0.5601 | 0.0115 [0.0058, 0.0170] |
| venus | {"c_H": 2, "c_W": 1, "c_R": 0.5} | 0.3822 | 0.4273 | 0.3901 | 0.4125 | 0.3896 | -0.0079 [-0.0139, -0.0016] |
| venus | {"c_H": 1, "c_W": 4, "c_R": 0.5} | 0.4355 | 0.9267 | 0.5079 | 0.4759 | 0.4293 | -0.0724 [-0.0912, -0.0520] |
