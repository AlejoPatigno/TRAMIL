# E1 rehearsal -- PILOT (plumbing check, not results)

PILOT: numbers are plumbing checks of the pipeline at rehearsal scale (few epochs, few replicates, a subset of the training bags). They are not results and support no claim about H1.

Bags: 48 (15 positive); config: {'folds': 2, 'repetition': 0, 'seeds': [11], 'max_epochs': 8}

| Arm | Fold AUROC (plumbing) | Mean | Selected epochs | Seconds |
| --- | --- | --- | --- | --- |
| residual | 0.723, 0.586 | 0.654 | [3, 1] | 3 |
| learned_only | 0.714, 0.672 | 0.693 | [2, 3] | 1 |
| abmil | 0.706, 0.656 | 0.681 | [4, 6] | 1 |
| gated_abmil | 0.756, 0.500 | 0.628 | [7, 1] | 2 |
| svm | 0.580, 0.578 | 0.579 | - | 1 |

Paired comparisons against the residual model (R = number of fold replicates; plumbing only):

- learned_only: mean diff -0.039, TOST 90% [-0.40352349949736815, 0.32598936084190594], p 0.624 (R=2)
- abmil: mean diff -0.027, TOST 90% [-0.3635878106005623, 0.31008203328963796], p 0.704 (R=2)
- gated_abmil: mean diff 0.026, TOST 90% [-0.436065849751643, 0.4883899043734917], p 0.782 (R=2)
- svm: mean diff 0.075, TOST 90% [-0.44679738237027616, 0.597467025227419], p 0.530 (R=2)

Run details:

- bags_requested: 48
- bags_used: 48
- instances_dropped_nonfinite: 1
- instances: 2804
- subset: stratified 48 of 141 training bags, seed 0
- run_seconds: 8.2
- extraction: 48 bags / 2805 instances in 2519.5 s wall (8 workers, 25 bags newly extracted, run overlapped with test runs); single-process cost measured 10.3 s per 205 s of audio per pass
- labels: PHQ-8 total >= 10 recomputed from detailed_lables.csv
- audio: E-DAIC instances_audio (offsets for 318/321/341/362 already applied by cut_instances.py)
