# Estimator A landmark initialization diagnostic

Validation: **PASS**. Saved seed 7, 500 steps, corrected noisy observations, nearest-neighbor association; no random data regenerated.

**48 estimated tracks; 8 unique true landmarks; 40 duplicate initializations.** A duplicate means a new track whose observation truth ID already has at least one earlier initialized track. Birth IDs are immutable evaluation labels, not proof that subsequent matches keep tracks unmixed.

| True landmark | Duplicate initializations |
|---|---:|
| L0 | 6 |
| L1 | 3 |
| L2 | 5 |
| L3 | 5 |
| L4 | 6 |
| L5 | 4 |
| L6 | 5 |
| L7 | 6 |

Duplicate step indices (zero-based): `[2, 3, 23, 24, 43, 48, 63, 64, 70, 84, 97, 98, 130, 133, 156, 204, 206, 242, 243, 253, 287, 291, 293, 306, 334, 347, 350, 367, 374, 379, 397, 405, 411, 423, 450, 454, 466, 480, 490]`.

## Confirmed observations

- The original NN accepts only squared Mahalanobis distance **strictly below 4** (not Euclidean distance, and not a squared threshold of 16). If no candidate passes, it returns -1 and the original augmentation immediately creates a permanent new track. No separate rejection or confirmation stage is used by this NN policy.
- Duplicate decision reasons: `{"all_candidate_squared_distances_at_or_above_gate": 40}`. The smallest candidate distance at duplicate events ranges from **4.041252 to 16.201845**.
- **0** duplicates lacked a same-birth-truth candidate at association time. This separates batch-order effects from rejection of already existing same-truth tracks.
- In **37** duplicate events, the nearest candidate itself had the correct birth truth ID but failed the gate.
- At duplicate events, **25** triggering observations have noise-only squared standardized norm >= 4, and **15** have norm < 4. This statistic uses stored noisy-minus-clean measurements divided by the configured measurement standard deviations. It is **not** the EKF association distance, which also includes state/map errors and innovation covariance.
- **43** existing-track matches disagree with birth truth IDs; **4** steps match multiple observations to the same track. NN considers each observation independently and does not enforce one-to-one matching. Full decision indices are in JSON.

The first incorrect existing-track match is `{'step_index': 22, 'observation_index': 0, 'truth_id': 4, 'assigned_track_id': 5, 'assigned_birth_truth_id': 5}`. Duplicate births at steps `[2, 3]` precede it, so wrong associations are not required to explain the onset of duplication.

## Interpretation: evidence versus hypotheses

The immediate confirmed mechanism is failure of the innovation gate followed by unconditional track creation. Repeated observations of an already mapped true landmark can therefore produce new tracks. The noisy measurement realizations, current predicted robot/map state, and their covariances jointly determine each gate failure; exact candidate residuals and innovation covariances are provided in JSON.

The most likely explanation is noisy measurements and noisy existing map/pose predictions crossing this fixed innovation gate, followed by immediate new-track creation. A noise-only norm below the gate does not imply the full innovation should pass: the prediction may already be biased. Incorrect matches can additionally contaminate later estimates, but this audit does not establish which earlier event caused each later failure. Isolating covariance calibration, linearization effects, noise outliers, map bias, or feedback from wrong matches requires controlled counterfactual experiments; none were performed and no threshold was tuned. For an ideal correctly normalized two-dimensional Gaussian innovation, this gate alone rejects exp(-4/2) = 13.53% of samples; the actual nonlinear correlated SLAM innovations need not follow that ideal model.

## Chronological initialization log

IDs and step indices below are zero-based. Time is after execution of command k, at (k+1)*0.1 s. Existing track IDs list all prior births of that truth ID, including earlier observations in this same augmentation batch. The JSON separately lists candidates available before batch correction/augmentation. Pose and triggering measurement are expanded below the table.

| Step | Time (s) | New track | True ID | Duplicate? | Existing same-truth tracks | Best candidate / squared distance | Reason |
|---:|---:|---|---|---|---|---|---|
| 0 | 0.1 | T0 | L3 | no | [] | none | no_existing_candidates |
| 0 | 0.1 | T1 | L2 | no | [] | none | no_existing_candidates |
| 2 | 0.3 | T2 | L3 | **YES** | [0] | T0/L3 / 6.636996 | all_candidate_squared_distances_at_or_above_gate |
| 3 | 0.4 | T3 | L2 | **YES** | [1] | T1/L2 / 6.429952 | all_candidate_squared_distances_at_or_above_gate |
| 8 | 0.9 | T4 | L4 | no | [] | T2/L3 / 174.854706 | all_candidate_squared_distances_at_or_above_gate |
| 20 | 2.1 | T5 | L5 | no | [] | T4/L4 / 10.517519 | all_candidate_squared_distances_at_or_above_gate |
| 23 | 2.4 | T6 | L4 | **YES** | [4] | T4/L4 / 4.212500 | all_candidate_squared_distances_at_or_above_gate |
| 24 | 2.5 | T7 | L5 | **YES** | [5] | T5/L5 / 4.523580 | all_candidate_squared_distances_at_or_above_gate |
| 40 | 4.1 | T8 | L7 | no | [] | T0/L3 / 115.364369 | all_candidate_squared_distances_at_or_above_gate |
| 40 | 4.1 | T9 | L6 | no | [] | T3/L2 / 59.184902 | all_candidate_squared_distances_at_or_above_gate |
| 43 | 4.4 | T10 | L6 | **YES** | [9] | T9/L6 / 4.139028 | all_candidate_squared_distances_at_or_above_gate |
| 48 | 4.9 | T11 | L7 | **YES** | [8] | T8/L7 / 6.951939 | all_candidate_squared_distances_at_or_above_gate |
| 62 | 6.3 | T12 | L0 | no | [] | T1/L2 / 38.757237 | all_candidate_squared_distances_at_or_above_gate |
| 63 | 6.4 | T13 | L0 | **YES** | [12] | T12/L0 / 4.494437 | all_candidate_squared_distances_at_or_above_gate |
| 64 | 6.5 | T14 | L7 | **YES** | [8, 11] | T8/L7 / 4.257146 | all_candidate_squared_distances_at_or_above_gate |
| 70 | 7.1 | T15 | L0 | **YES** | [12, 13] | T13/L0 / 5.618746 | all_candidate_squared_distances_at_or_above_gate |
| 75 | 7.6 | T16 | L1 | no | [] | T3/L2 / 6.648800 | all_candidate_squared_distances_at_or_above_gate |
| 84 | 8.5 | T17 | L1 | **YES** | [16] | T16/L1 / 10.123226 | all_candidate_squared_distances_at_or_above_gate |
| 97 | 9.8 | T18 | L3 | **YES** | [0, 2] | T0/L3 / 4.251716 | all_candidate_squared_distances_at_or_above_gate |
| 98 | 9.9 | T19 | L3 | **YES** | [0, 2, 18] | T2/L3 / 4.783503 | all_candidate_squared_distances_at_or_above_gate |
| 130 | 13.1 | T20 | L4 | **YES** | [4, 6] | T4/L4 / 4.213225 | all_candidate_squared_distances_at_or_above_gate |
| 130 | 13.1 | T21 | L5 | **YES** | [5, 7] | T7/L5 / 5.079635 | all_candidate_squared_distances_at_or_above_gate |
| 133 | 13.4 | T22 | L5 | **YES** | [5, 7, 21] | T7/L5 / 6.130591 | all_candidate_squared_distances_at_or_above_gate |
| 156 | 15.7 | T23 | L7 | **YES** | [8, 11, 14] | T11/L7 / 6.428634 | all_candidate_squared_distances_at_or_above_gate |
| 204 | 20.5 | T24 | L2 | **YES** | [1, 3] | T1/L2 / 4.134777 | all_candidate_squared_distances_at_or_above_gate |
| 206 | 20.7 | T25 | L4 | **YES** | [4, 6, 20] | T6/L4 / 5.361797 | all_candidate_squared_distances_at_or_above_gate |
| 242 | 24.3 | T26 | L6 | **YES** | [9, 10] | T10/L6 / 10.885918 | all_candidate_squared_distances_at_or_above_gate |
| 243 | 24.4 | T27 | L6 | **YES** | [9, 10, 26] | T9/L6 / 4.882665 | all_candidate_squared_distances_at_or_above_gate |
| 253 | 25.4 | T28 | L6 | **YES** | [9, 10, 26, 27] | T9/L6 / 6.510422 | all_candidate_squared_distances_at_or_above_gate |
| 287 | 28.8 | T29 | L0 | **YES** | [12, 13, 15] | T12/L0 / 4.156947 | all_candidate_squared_distances_at_or_above_gate |
| 291 | 29.2 | T30 | L2 | **YES** | [1, 3, 24] | T1/L2 / 8.076185 | all_candidate_squared_distances_at_or_above_gate |
| 293 | 29.4 | T31 | L2 | **YES** | [1, 3, 24, 30] | T24/L2 / 4.624320 | all_candidate_squared_distances_at_or_above_gate |
| 306 | 30.7 | T32 | L3 | **YES** | [0, 2, 18, 19] | T19/L3 / 4.584328 | all_candidate_squared_distances_at_or_above_gate |
| 334 | 33.5 | T33 | L4 | **YES** | [4, 6, 20, 25] | T4/L4 / 4.276073 | all_candidate_squared_distances_at_or_above_gate |
| 347 | 34.8 | T34 | L7 | **YES** | [8, 11, 14, 23] | T14/L7 / 5.413410 | all_candidate_squared_distances_at_or_above_gate |
| 350 | 35.1 | T35 | L7 | **YES** | [8, 11, 14, 23, 34] | T34/L7 / 4.574344 | all_candidate_squared_distances_at_or_above_gate |
| 367 | 36.8 | T36 | L0 | **YES** | [12, 13, 15, 29] | T15/L0 / 4.041252 | all_candidate_squared_distances_at_or_above_gate |
| 374 | 37.5 | T37 | L0 | **YES** | [12, 13, 15, 29, 36] | T15/L0 / 5.170588 | all_candidate_squared_distances_at_or_above_gate |
| 379 | 38.0 | T38 | L1 | **YES** | [16, 17] | T13/L0 / 4.507048 | all_candidate_squared_distances_at_or_above_gate |
| 397 | 39.8 | T39 | L2 | **YES** | [1, 3, 24, 30, 31] | T24/L2 / 4.821175 | all_candidate_squared_distances_at_or_above_gate |
| 405 | 40.6 | T40 | L4 | **YES** | [4, 6, 20, 25, 33] | T32/L3 / 16.201845 | all_candidate_squared_distances_at_or_above_gate |
| 411 | 41.2 | T41 | L4 | **YES** | [4, 6, 20, 25, 33, 40] | T5/L5 / 4.406276 | all_candidate_squared_distances_at_or_above_gate |
| 423 | 42.4 | T42 | L5 | **YES** | [5, 7, 21, 22] | T22/L5 / 4.480937 | all_candidate_squared_distances_at_or_above_gate |
| 450 | 45.1 | T43 | L6 | **YES** | [9, 10, 26, 27, 28] | T26/L6 / 4.282044 | all_candidate_squared_distances_at_or_above_gate |
| 454 | 45.5 | T44 | L7 | **YES** | [8, 11, 14, 23, 34, 35] | T11/L7 / 5.929265 | all_candidate_squared_distances_at_or_above_gate |
| 466 | 46.7 | T45 | L0 | **YES** | [12, 13, 15, 29, 36, 37] | T36/L0 / 4.860748 | all_candidate_squared_distances_at_or_above_gate |
| 480 | 48.1 | T46 | L1 | **YES** | [16, 17, 38] | T16/L1 / 4.265699 | all_candidate_squared_distances_at_or_above_gate |
| 490 | 49.1 | T47 | L3 | **YES** | [0, 2, 18, 19, 32] | T2/L3 / 4.566819 | all_candidate_squared_distances_at_or_above_gate |

## First duplicate example

At step 2 (t=0.3 s), T2 duplicates L3. Existing T0 has d_squared=6.636996 and innovation `[range=-11.145887, bearing=0.524552]`; the recorded bearing innovation standard deviation is 0.217983 rad. T0's initial observation had bearing noise -0.306257 rad, while the duplicate-triggering observation has bearing noise +0.187181 rad and noise-only squared norm 1.701578. These are measured facts showing prediction/measurement disagreement; they do not by themselves isolate one cause.


## Initialization details

Pose is [x, y, theta] after the step's existing-landmark correction, which is the pose actually used for augmentation. The separate association pose in JSON is the predicted pose before correction. Range units are simulator coordinates; bearing and heading are radians. Full candidate tables include all gated-out landmarks, the best candidate sharing the true ID, innovations and S.

- Step 0, t=0.1, T0/L3 first initialization: pose `[190.0, 50.0, 0.0]`; measurement `[range=257.298150843, bearing=-0.506072630]`.
- Step 0, t=0.1, T1/L2 first initialization: pose `[190.0, 50.0, 0.0]`; measurement `[range=128.730818869, bearing=-0.533221663]`.
- Step 2, t=0.3, T2/L3 **DUPLICATE**: pose `[210.01159677335195, 50.00163498401998, 0.0008811649371899012]`; measurement `[range=229.423515734, bearing=0.012473873]`.
- Step 3, t=0.4, T3/L2 **DUPLICATE**: pose `[220.00251534289984, 49.98057699489302, -0.0014583894372165308]`; measurement `[range=76.605743266, bearing=-0.438288766]`.
- Step 8, t=0.9, T4/L4 first initialization: pose `[270.06746347839896, 49.199921960142085, -0.013457179384791922]`; measurement `[range=295.371051599, bearing=1.220817737]`.
- Step 20, t=2.1, T5/L5 first initialization: pose `[387.07792594772167, 55.21809186347289, 1.5696076105841195]`; measurement `[range=241.525804031, bearing=0.330908665]`.
- Step 23, t=2.4, T6/L4 **DUPLICATE**: pose `[387.455111857253, 85.85230146936362, 1.5590102143568016]`; measurement `[range=200.597208542, bearing=-0.269855762]`.
- Step 24, t=2.5, T7/L5 **DUPLICATE**: pose `[387.57643530488235, 95.8345843714975, 1.558277298407862]`; measurement `[range=218.662361077, bearing=0.729050786]`.
- Step 40, t=4.1, T8/L7 first initialization: pose `[393.87173175707323, 246.36693326412643, 3.103706401410502]`; measurement `[range=371.450355868, bearing=-0.094021001]`.
- Step 40, t=4.1, T9/L6 first initialization: pose `[393.87173175707323, 246.36693326412643, 3.103706401410502]`; measurement `[range=240.255129613, bearing=-0.253656833]`.
- Step 43, t=4.4, T10/L6 **DUPLICATE**: pose `[365.21222786838166, 247.37530903191586, 3.0898553218890052]`; measurement `[range=223.047314971, bearing=0.038532356]`.
- Step 48, t=4.9, T11/L7 **DUPLICATE**: pose `[315.48520961380177, 249.8965391824505, 3.091453273851465]`; measurement `[range=316.526356537, bearing=0.077274653]`.
- Step 62, t=6.3, T12/L0 first initialization: pose `[172.4464999758866, 250.05976070063699, -3.142495523489264]`; measurement `[range=290.723425940, bearing=0.907709187]`.
- Step 63, t=6.4, T13/L0 **DUPLICATE**: pose `[161.90359578104733, 248.95067921410907, 3.1500757952482332]`; measurement `[range=303.507472036, bearing=1.370368649]`.
- Step 64, t=6.5, T14/L7 **DUPLICATE**: pose `[151.84494335665232, 248.71104633076726, -3.1333576141844643]`; measurement `[range=124.995079298, bearing=-0.096799195]`.
- Step 70, t=7.1, T15/L0 **DUPLICATE**: pose `[94.90729401808146, 241.2610162974549, -1.5774557440311716]`; measurement `[range=286.525913432, bearing=-0.548632078]`.
- Step 75, t=7.6, T16/L1 first initialization: pose `[93.89763951920482, 189.51251769337404, -1.5785448768414525]`; measurement `[range=221.249560965, bearing=0.603946687]`.
- Step 84, t=8.5, T17/L1 **DUPLICATE**: pose `[96.01942178497704, 93.77407186660055, -1.5546299538946204]`; measurement `[range=115.519631233, bearing=0.781999578]`.
- Step 97, t=9.8, T18/L3 **DUPLICATE**: pose `[156.59333155466754, 49.030782275726466, -0.01200324091556134]`; measurement `[range=298.542771926, bearing=-0.631818884]`.
- Step 98, t=9.9, T19/L3 **DUPLICATE**: pose `[166.2134234500678, 48.949514391341665, -0.017350077489554187]`; measurement `[range=298.009969670, bearing=0.048607455]`.
- Step 130, t=13.1, T20/L4 **DUPLICATE**: pose `[385.8045806707251, 149.92780034718885, 1.5808890917926952]`; measurement `[range=175.456186705, bearing=-0.047785873]`.
- Step 130, t=13.1, T21/L5 **DUPLICATE**: pose `[385.8045806707251, 149.92780034718885, 1.5808890917926952]`; measurement `[range=193.103339143, bearing=0.314611702]`.
- Step 133, t=13.4, T22/L5 **DUPLICATE**: pose `[386.85487548605954, 179.48405542996807, 1.5623180370178742]`; measurement `[range=158.436571722, bearing=1.294841401]`.
- Step 156, t=15.7, T23/L7 **DUPLICATE**: pose `[227.89095446446822, 239.84648595301655, -3.1272694830946994]`; measurement `[range=258.173357660, bearing=0.287616624]`.
- Step 204, t=20.5, T24/L2 **DUPLICATE**: pose `[238.89308965487407, 43.04626638727801, 0.028856605192079898]`; measurement `[range=84.510584664, bearing=-1.077944970]`.
- Step 206, t=20.7, T25/L4 **DUPLICATE**: pose `[259.0291187639976, 45.09069107851813, 0.031077343546792948]`; measurement `[range=282.285564577, bearing=0.678678660]`.
- Step 242, t=24.3, T26/L6 **DUPLICATE**: pose `[337.0558775829009, 253.9068637261799, -2.9423770639159867]`; measurement `[range=230.400497849, bearing=-0.659955977]`.
- Step 243, t=24.4, T27/L6 **DUPLICATE**: pose `[327.38310457153284, 252.69473975775423, -2.967076844356298]`; measurement `[range=177.233799416, bearing=0.066247551]`.
- Step 253, t=25.4, T28/L6 **DUPLICATE**: pose `[228.91935063078614, 240.4749795483451, -3.01288573686174]`; measurement `[range=80.099949198, bearing=-1.037948350]`.
- Step 287, t=28.8, T29/L0 **DUPLICATE**: pose `[71.50988130583616, 46.31205313444445, -1.516091322481365]`; measurement `[range=90.189352442, bearing=-0.333690677]`.
- Step 291, t=29.2, T30/L2 **DUPLICATE**: pose `[84.61357140361699, 26.598339904779255, 0.06283136068589931]`; measurement `[range=243.009492304, bearing=0.226426487]`.
- Step 293, t=29.4, T31/L2 **DUPLICATE**: pose `[104.45768757349336, 27.501329711820343, 0.05435848225077709]`; measurement `[range=211.048832646, bearing=-0.643486572]`.
- Step 306, t=30.7, T32/L3 **DUPLICATE**: pose `[230.8120627998544, 37.4435585133265, 0.07278714989889519]`; measurement `[range=218.196276321, bearing=0.259162898]`.
- Step 334, t=33.5, T33/L4 **DUPLICATE**: pose `[356.7355231277456, 189.60063279666383, 1.6445331321131242]`; measurement `[range=139.232729898, bearing=-1.110100444]`.
- Step 347, t=34.8, T34/L7 **DUPLICATE**: pose `[279.97151168517934, 229.95540861453009, -3.0094464828064877]`; measurement `[range=275.614234579, bearing=0.174839108]`.
- Step 350, t=35.1, T35/L7 **DUPLICATE**: pose `[249.6386029680856, 225.61059917170945, -3.0010104912541546]`; measurement `[range=221.615561544, bearing=-0.030214886]`.
- Step 367, t=36.8, T36/L0 **DUPLICATE**: pose `[78.50515573861222, 214.65632073631502, -3.0787732496638456]`; measurement `[range=246.457800058, bearing=0.892422783]`.
- Step 374, t=37.5, T37/L0 **DUPLICATE**: pose `[51.44187859426394, 170.17409161507908, -1.5457848792120135]`; measurement `[range=225.053562807, bearing=-0.619946974]`.
- Step 379, t=38.0, T38/L1 **DUPLICATE**: pose `[53.979937113966706, 120.42270840550118, -1.5301793327980855]`; measurement `[range=196.024506824, bearing=0.233892978]`.
- Step 397, t=39.8, T39/L2 **DUPLICATE**: pose `[130.72769538153042, 20.284486179415357, 0.0037284485723334234]`; measurement `[range=144.170510894, bearing=-0.645218230]`.
- Step 405, t=40.6, T40/L4 **DUPLICATE**: pose `[211.45536184168168, 23.95547595496902, 0.019260223390059273]`; measurement `[range=292.114254220, bearing=0.411792897]`.
- Step 411, t=41.2, T41/L4 **DUPLICATE**: pose `[271.75861824695625, 26.642127294543226, 0.04841747003663964]`; measurement `[range=258.375733984, bearing=1.107327439]`.
- Step 423, t=42.4, T42/L5 **DUPLICATE**: pose `[361.9442331853577, 70.39875230551031, 1.5459195760306848]`; measurement `[range=193.857918506, bearing=0.409617369]`.
- Step 450, t=45.1, T43/L6 **DUPLICATE**: pose `[271.46750729170486, 245.42248974578953, 3.008359817423139]`; measurement `[range=150.009617263, bearing=-0.546854469]`.
- Step 454, t=45.5, T44/L7 **DUPLICATE**: pose `[231.25907302408365, 252.0469799027217, 2.9991924897919304]`; measurement `[range=251.162760155, bearing=-0.227096698]`.
- Step 466, t=46.7, T45/L0 **DUPLICATE**: pose `[111.95696869119836, 264.7303579855469, 3.0237585443362]`; measurement `[range=311.214369757, bearing=0.880700198]`.
- Step 480, t=48.1, T46/L1 **DUPLICATE**: pose `[77.55233011446762, 156.27719964438975, -1.547093165604845]`; measurement `[range=188.216517420, bearing=0.839856818]`.
- Step 490, t=49.1, T47/L3 **DUPLICATE**: pose `[76.82517858890294, 66.64454701960065, -0.007983667800362939]`; measurement `[range=348.570868339, bearing=-0.423187877]`.

## Evidence and validation

The original saved NPZ replay contains posterior states and final decisions, not pre-correction candidate innovation statistics. This tool uses the existing replay harness, wrapping only the NN callable and observing its actual local variables with a temporary Python trace. It does not recalculate candidate distances using posterior poses, reimplement association, or modify the original function's code, thresholds or return arrays. The previous trace hook and original callable are restored. Run outside a debugger/coverage session to avoid interfering with their trace events.

Without an instrumented replay, candidate statistics must be marked unavailable. Long-term audit logs would need the pre-association state/covariance, consumed measurement, candidate H/r/S/distance and gate, or the live candidate values captured here. The initialization collector supports unavailable decision evidence explicitly.

Validation checks: `{"association_callable_and_threshold_restored": true, "diagnostic_replay_matches_saved_numerical_arrays": true, "diagnostics_on_off_numerical_arrays_exact": true, "legacy_source_and_inputs_unchanged": true, "plain_replay_matches_saved_numerical_arrays": true, "reproduced_40_duplicates": true, "reproduced_48_tracks": true, "reproduced_8_unique_truth_ids": true, "source_files_unchanged": true}`.

All numerical state, covariance, landmark, observation and association arrays are compared exactly against diagnostics-disabled replay and the original saved output (NaN padding equal). Timing arrays are excluded: tracing adds overhead, and no diagnostic timing replaces the baseline benchmark timings. Original source and input archives are hash-checked for preservation. JSON contains source hashes, dependency versions, the full event log and all 1,000 association decisions.

Reproduce from the repository root:

```sh
python3 -B -m benchmarks.diagnose_initializations
python3 -B -m pytest -q tests/test_diagnose_initializations.py
```

The existing animation already highlights duplicate births in magenta and supports Next issue / N. Use `python3 -B -m benchmarks.visualize_estimator_a --mode noisy --association nn --backend QtAgg --frame 2` to inspect the first duplicate frame. No visualization or estimator changes are made by this diagnostic.
